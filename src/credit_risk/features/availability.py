"""Feature availability matrix: when does each column exist?

The rule the whole project depends on: a feature may only use information that
exists at the observation date. configs/feature_availability.yaml states, for
every column, which class it belongs to and whether it may be modelled.

Resolution order is columns -> patterns -> table default. A column that matches
no rule is an error, not a silent pass: a column that appears in the data later
has to be classified by a human before anything can model it.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

AVAILABILITY_CLASSES = {"at_application", "historical", "post_decision", "identifier", "target"}
USABLE_CLASSES = {"at_application", "historical"}

_RULE_KEYS = {"availability", "use", "caveat"}
_TABLE_KEYS = {"grain", "role", "evidence", "default", "columns", "patterns", "row_filter", "use_for_features"}


class AvailabilityError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    availability: str
    use: bool
    caveat: str | None = None

    @classmethod
    def parse(cls, where: str, raw: Any) -> Rule:
        if not isinstance(raw, dict):
            raise AvailabilityError(f"{where}: rule must be a mapping, got {raw!r}")
        unknown = set(raw) - _RULE_KEYS
        if unknown:
            raise AvailabilityError(f"{where}: unknown keys {sorted(unknown)}")
        availability = raw.get("availability")
        if availability not in AVAILABILITY_CLASSES:
            raise AvailabilityError(
                f"{where}: availability must be one of {sorted(AVAILABILITY_CLASSES)}, got {availability!r}"
            )
        use = raw.get("use", availability in USABLE_CLASSES)
        if not isinstance(use, bool):
            raise AvailabilityError(f"{where}: use must be true or false, got {use!r}")
        if use and availability not in USABLE_CLASSES:
            raise AvailabilityError(f"{where}: a {availability} column can never be used as a feature")
        return cls(availability, use, raw.get("caveat"))


@dataclass(frozen=True)
class ColumnAvailability:
    table: str
    column: str
    availability: str
    use: bool
    caveat: str | None
    matched_by: str  # column | pattern:<glob> | default


@dataclass(frozen=True)
class TableAvailability:
    name: str
    grain: str
    default: Rule
    columns: dict[str, Rule] = field(default_factory=dict)
    patterns: tuple[tuple[str, Rule], ...] = ()
    evidence: str | None = None
    role: str | None = None
    row_filter: str | None = None
    use_for_features: bool = True

    def resolve(self, column: str) -> ColumnAvailability:
        rule, matched_by = self.columns.get(column), "column"
        if rule is None:
            for glob, pattern_rule in self.patterns:
                if fnmatch.fnmatchcase(column, glob):
                    rule, matched_by = pattern_rule, f"pattern:{glob}"
                    break
        if rule is None:
            rule, matched_by = self.default, "default"
        use = rule.use and self.use_for_features
        return ColumnAvailability(self.name, column, rule.availability, use, rule.caveat, matched_by)

    @classmethod
    def parse(cls, source: str, name: str, raw: dict[str, Any]) -> TableAvailability:
        where = f"{source}.{name}"
        if not isinstance(raw, dict):
            raise AvailabilityError(f"{where}: table must be a mapping")
        unknown = set(raw) - _TABLE_KEYS
        if unknown:
            raise AvailabilityError(f"{where}: unknown keys {sorted(unknown)}")
        if "default" not in raw:
            raise AvailabilityError(f"{where}: a default rule is required, so no column is left unclassified")

        patterns = []
        for entry in raw.get("patterns") or []:
            if "match" not in entry:
                raise AvailabilityError(f"{where}: every pattern needs a 'match' glob")
            glob = entry["match"]
            patterns.append((glob, Rule.parse(f"{where} pattern {glob}", {k: v for k, v in entry.items() if k != "match"})))

        return cls(
            name=name,
            grain=raw.get("grain", ""),
            default=Rule.parse(f"{where} default", raw["default"]),
            columns={
                col: Rule.parse(f"{where}.{col}", rule) for col, rule in (raw.get("columns") or {}).items()
            },
            patterns=tuple(patterns),
            evidence=raw.get("evidence"),
            role=raw.get("role"),
            row_filter=raw.get("row_filter"),
            use_for_features=bool(raw.get("use_for_features", True)),
        )


@dataclass(frozen=True)
class AvailabilityMatrix:
    source: str
    tables: dict[str, TableAvailability]

    def table(self, name: str) -> TableAvailability:
        if name not in self.tables:
            raise AvailabilityError(
                f"{self.source}.{name} has no availability rules; classify it in "
                "configs/feature_availability.yaml before using it"
            )
        return self.tables[name]

    def classify(self, table: str, columns: list[str]) -> list[ColumnAvailability]:
        spec = self.table(table)
        return [spec.resolve(column) for column in columns]

    def feature_columns(self, table: str, columns: list[str]) -> list[str]:
        return [c.column for c in self.classify(table, columns) if c.use]


def parse_availability(raw: dict[str, Any], source: str) -> AvailabilityMatrix:
    if source not in raw:
        raise AvailabilityError(f"feature_availability.yaml has no section for {source}")
    tables = {
        name: TableAvailability.parse(source, name, spec)
        for name, spec in raw[source].items()
    }
    return AvailabilityMatrix(source, tables)


def availability_frame(matrix: AvailabilityMatrix, columns_by_table: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        {
            "table": c.table,
            "column": c.column,
            "availability": c.availability,
            "use": c.use,
            "matched_by": c.matched_by,
            "caveat": c.caveat or "",
        }
        for table, columns in columns_by_table.items()
        for c in matrix.classify(table, columns)
    ]
    return pd.DataFrame(rows, columns=["table", "column", "availability", "use", "matched_by", "caveat"])


def availability_markdown(matrix: AvailabilityMatrix, frame: pd.DataFrame) -> str:
    lines = [
        f"# Feature availability matrix - {matrix.source}",
        "",
        f"{len(frame)} cột, {int(frame['use'].sum())} dùng được làm feature.",
        "",
        "| Bảng | Lớp | Số cột | Dùng được |",
        "|---|---|---:|---:|",
    ]
    grouped = frame.groupby(["table", "availability"], sort=False)
    for (table, availability), part in grouped:
        lines.append(f"| {table} | {availability} | {len(part)} | {int(part['use'].sum())} |")

    for table in frame["table"].unique():
        spec = matrix.table(table)
        part = frame[frame["table"] == table]
        lines += ["", f"## {table}", "", f"*{spec.grain}*", ""]
        if spec.role:
            lines.append(f"- Vai trò: `{spec.role}`")
        if not spec.use_for_features:
            lines.append("- **Không dùng để tạo feature.**")
        if spec.row_filter:
            lines.append(f"- Lọc dòng khi tổng hợp: `{spec.row_filter}`")
        if spec.evidence:
            lines.append(f"- Kiểm chứng: {spec.evidence}")
        lines += ["", "| Cột | Lớp | Dùng | Khớp theo | Lưu ý |", "|---|---|---|---|---|"]
        for row in part.itertuples(index=False):
            lines.append(
                f"| `{row.column}` | {row.availability} | {'có' if row.use else 'không'} "
                f"| {row.matched_by} | {row.caveat} |"
            )
    return "\n".join(lines) + "\n"
