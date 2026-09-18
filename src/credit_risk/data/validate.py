"""Step 2 - validate-data: check processed parquet against the data contract.

Every rule produces one CheckResult. The source passes only if no rule with
severity "error" fails; "warn" rules are reported but never block the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from credit_risk.data.contracts import ColumnSpec, SourceContract, TableSpec
from credit_risk.data.db import sql_ident, sql_str
from credit_risk.data.ingest import processed_path
from credit_risk.settings import Settings

INTEGER_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                 "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}
FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}


@dataclass(frozen=True)
class CheckResult:
    table: str
    check: str
    column: str | None
    severity: str
    passed: bool
    n_failed: int
    detail: str


@dataclass
class ValidationReport:
    source: str
    results: list[CheckResult]

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warn"]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "passed": self.passed,
            "n_checks": len(self.results),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Data validation - {self.source}: {verdict}",
            "",
            f"{len(self.results)} checks, {len(self.errors)} errors, {len(self.warnings)} warnings.",
            "",
            "| Table | Check | Column | Severity | Result | Failed rows | Detail |",
            "|---|---|---|---|---|---|---|",
        ]
        ordered = sorted(self.results, key=lambda r: (r.passed, r.severity != "error", r.table))
        for r in ordered:
            result = "pass" if r.passed else "FAIL"
            lines.append(
                f"| {r.table} | {r.check} | {r.column or ''} | {r.severity} | {result} "
                f"| {r.n_failed:,} | {r.detail} |"
            )
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{self.source}.json"
        md_path = out_dir / f"{self.source}.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return [json_path, md_path]


def _dtype_matches(expected: str, actual: str) -> bool:
    base = actual.split("(")[0].upper()
    if expected == "integer":
        return base in INTEGER_TYPES
    if expected == "number":
        return base in INTEGER_TYPES or base in FLOAT_TYPES or base == "DECIMAL"
    if expected == "string":
        return base == "VARCHAR"
    return base == "DATE"  # date, yyyymm


def _literal(value: object) -> str:
    return str(value) if isinstance(value, (int, float)) else sql_str(str(value))


class _TableChecker:
    def __init__(self, con: duckdb.DuckDBPyConnection, table: TableSpec, path: Path):
        self.con = con
        self.table = table
        self.rel = f"read_parquet({sql_str(path)})"
        self.results: list[CheckResult] = []

    def scalar(self, sql: str) -> int:
        return int(self.con.execute(sql).fetchone()[0] or 0)

    def add(self, check: str, column: str | None, severity: str, n_failed: int, detail: str) -> None:
        self.results.append(
            CheckResult(self.table.name, check, column, severity, n_failed == 0, n_failed, detail)
        )

    def run(self, all_paths: dict[str, Path]) -> list[CheckResult]:
        n_rows = self.scalar(f"SELECT COUNT(*) FROM {self.rel}")
        shortfall = max(self.table.min_rows - n_rows, 0)
        self.add("min_rows", None, "error", shortfall, f"{n_rows:,} rows, minimum {self.table.min_rows:,}")

        actual_types = {
            row[0]: row[1] for row in self.con.execute(f"DESCRIBE SELECT * FROM {self.rel}").fetchall()
        }
        present: list[ColumnSpec] = []
        for col in self.table.columns:
            if col.name not in actual_types:
                self.add("required_column", col.name, "error", 1, "column missing")
                continue
            present.append(col)
            # Keys and targets must have the right type; other columns follow their own severity.
            severity = "error" if col.role in {"key", "target"} else col.severity
            ok = _dtype_matches(col.dtype, actual_types[col.name])
            self.add("dtype", col.name, severity, 0 if ok else 1,
                     f"expected {col.dtype}, found {actual_types[col.name]}")

        present_names = {c.name for c in present}
        if self.table.primary_key and set(self.table.primary_key) <= present_names:
            self._check_primary_key()
        for col in present:
            self._check_column(col, n_rows)
        for fk in self.table.foreign_keys:
            if fk.column in present_names:
                self._check_foreign_key(fk, all_paths)
        return self.results

    def _check_primary_key(self) -> None:
        keys = ", ".join(sql_ident(k) for k in self.table.primary_key)
        n_dup = self.scalar(
            f"SELECT COUNT(*) FROM (SELECT {keys} FROM {self.rel} GROUP BY ALL HAVING COUNT(*) > 1)"
        )
        self.add("unique_key", "+".join(self.table.primary_key), self.table.primary_key_severity,
                 n_dup, f"{n_dup:,} key values appear more than once")

    def _check_column(self, col: ColumnSpec, n_rows: int) -> None:
        c = sql_ident(col.name)
        n_null = self.scalar(f"SELECT COUNT(*) FROM {self.rel} WHERE {c} IS NULL")
        if not col.nullable:
            severity = "error" if col.role in {"key", "target"} else col.severity
            self.add("not_null", col.name, severity, n_null, f"{n_null:,} null values")
        if col.max_missing is not None:
            rate = n_null / n_rows if n_rows else 0.0
            self.add("max_missing", col.name, col.severity, n_null if rate > col.max_missing else 0,
                     f"missing rate {rate:.4f}, maximum {col.max_missing}")
        if col.allowed_values is not None:
            allowed = ", ".join(sql_str(v) for v in col.allowed_values)
            n_bad = self.scalar(
                f"SELECT COUNT(*) FROM {self.rel} WHERE {c} IS NOT NULL AND CAST({c} AS VARCHAR) NOT IN ({allowed})"
            )
            self.add("allowed_values", col.name, col.severity, n_bad, f"allowed {list(col.allowed_values)}")
        if col.min is not None or col.max is not None:
            conds = []
            if col.min is not None:
                conds.append(f"{c} < {col.min}")
            if col.max is not None:
                conds.append(f"{c} > {col.max}")
            sentinel = ""
            if col.sentinel_values:
                sentinel = f" AND {c} NOT IN ({', '.join(_literal(v) for v in col.sentinel_values)})"
            n_bad = self.scalar(
                f"SELECT COUNT(*) FROM {self.rel} WHERE {c} IS NOT NULL{sentinel} AND ({' OR '.join(conds)})"
            )
            self.add("range", col.name, col.severity, n_bad,
                     f"expected [{col.min}, {col.max}], sentinels {list(col.sentinel_values)}")
        if col.pattern is not None:
            n_bad = self.scalar(
                f"SELECT COUNT(*) FROM {self.rel} WHERE {c} IS NOT NULL "
                f"AND NOT regexp_full_match({c}, {sql_str(col.pattern)})"
            )
            self.add("pattern", col.name, col.severity, n_bad, f"pattern {col.pattern}")

    def _check_foreign_key(self, fk, all_paths: dict[str, Path]) -> None:
        c = sql_ident(fk.column)
        refs = [(t, col) for t, col in fk.references if t in all_paths and all_paths[t].exists()]
        if not refs:
            self.add("foreign_key", fk.column, fk.severity, 1, "referenced table not available")
            return
        union = " UNION ".join(
            f"SELECT {sql_ident(col)} AS k FROM read_parquet({sql_str(all_paths[t])})" for t, col in refs
        )
        # ANTI JOIN rather than NOT IN: a NULL in the referenced column would make NOT IN match nothing.
        n_orphan = self.scalar(
            f"SELECT COUNT(*) FROM {self.rel} AS t ANTI JOIN ({union}) AS u ON t.{c} = u.k "
            f"WHERE t.{c} IS NOT NULL"
        )
        targets = [f"{t}.{col}" for t, col in refs]
        self.add("foreign_key", fk.column, fk.severity, n_orphan, f"{n_orphan:,} rows without match in {targets}")


def validate_source(
    contract: SourceContract, settings: Settings, con: duckdb.DuckDBPyConnection
) -> ValidationReport:
    paths = {name: processed_path(contract, t, settings) for name, t in contract.tables.items()}
    results: list[CheckResult] = []
    for name, table in contract.tables.items():
        if not paths[name].exists():
            results.append(CheckResult(name, "table_exists", None, "error", False, 1,
                                       f"{paths[name].name} not found - run ingest first"))
            continue
        results.extend(_TableChecker(con, table, paths[name]).run(paths))
    return ValidationReport(contract.source, results)
