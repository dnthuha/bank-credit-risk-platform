"""Typed view of configs/data_contracts/*.yaml.

Parsing is strict: an unknown key or dtype is a ConfigError, so a typo in a
contract cannot silently disable a rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from credit_risk.config import ConfigError

DTYPES = {"integer", "number", "string", "date", "yyyymm"}
SEVERITIES = {"error", "warn"}
FORMATS = {"csv", "pipe_no_header"}

_COLUMN_KEYS = {
    "name", "dtype", "role", "nullable", "max_missing", "allowed_values", "min", "max",
    "sentinel_values", "na_values", "pattern", "severity", "description",
}
_TABLE_KEYS = {
    "file", "file_glob", "description", "primary_key", "primary_key_severity",
    "min_rows", "foreign_keys", "columns",
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    role: str | None = None
    nullable: bool = True
    max_missing: float | None = None
    allowed_values: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    sentinel_values: tuple[Any, ...] = ()
    na_values: tuple[str, ...] = ()
    pattern: str | None = None
    severity: str = "error"
    description: str | None = None


@dataclass(frozen=True)
class ForeignKey:
    column: str
    references: tuple[tuple[str, str], ...]  # (table, column)
    severity: str = "error"


@dataclass(frozen=True)
class TableSpec:
    name: str
    description: str
    columns: tuple[ColumnSpec, ...]
    file: str | None = None
    file_glob: str | None = None
    primary_key: tuple[str, ...] = ()
    primary_key_severity: str = "error"
    min_rows: int = 1
    foreign_keys: tuple[ForeignKey, ...] = ()

    def column(self, name: str) -> ColumnSpec:
        return next(c for c in self.columns if c.name == name)


@dataclass(frozen=True)
class SourceContract:
    source: str
    raw_subdir: str
    format: str
    tables: dict[str, TableSpec]
    layout_version: str | None = None


def _check_keys(where: str, given: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(given) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")


def _check_severity(where: str, value: str) -> str:
    if value not in SEVERITIES:
        raise ConfigError(f"{where}: severity must be one of {sorted(SEVERITIES)}, got {value!r}")
    return value


def _parse_column(where: str, raw: dict[str, Any]) -> ColumnSpec:
    _check_keys(where, raw, _COLUMN_KEYS)
    if raw.get("dtype") not in DTYPES:
        raise ConfigError(f"{where}: dtype must be one of {sorted(DTYPES)}")
    allowed = raw.get("allowed_values")
    return ColumnSpec(
        name=raw["name"],
        dtype=raw["dtype"],
        role=raw.get("role"),
        nullable=bool(raw.get("nullable", True)),
        max_missing=raw.get("max_missing"),
        allowed_values=tuple(str(v) for v in allowed) if allowed is not None else None,
        min=raw.get("min"),
        max=raw.get("max"),
        sentinel_values=tuple(raw.get("sentinel_values", ())),
        na_values=tuple(str(v) for v in raw.get("na_values", ())),
        pattern=raw.get("pattern"),
        severity=_check_severity(where, raw.get("severity", "error")),
        description=raw.get("description"),
    )


def _parse_table(source: str, name: str, raw: dict[str, Any]) -> TableSpec:
    where = f"{source}.{name}"
    _check_keys(where, raw, _TABLE_KEYS)
    if bool(raw.get("file")) == bool(raw.get("file_glob")):
        raise ConfigError(f"{where}: exactly one of file / file_glob is required")

    raw_columns = raw.get("columns") or []
    if isinstance(raw_columns, dict):  # mapping form: {COL: {...}}
        raw_columns = [{"name": col, **spec} for col, spec in raw_columns.items()]
    columns = tuple(_parse_column(f"{where}.{c.get('name')}", c) for c in raw_columns)
    names = [c.name for c in columns]
    if len(names) != len(set(names)):
        raise ConfigError(f"{where}: duplicate column names")

    primary_key = tuple(raw.get("primary_key", ()))
    missing_pk = set(primary_key) - set(names)
    if missing_pk:
        raise ConfigError(f"{where}: primary key columns not declared: {sorted(missing_pk)}")

    foreign_keys = []
    for fk in raw.get("foreign_keys", []):
        refs = tuple(tuple(ref.split(".", 1)) for ref in fk["references"])
        foreign_keys.append(
            ForeignKey(fk["column"], refs, _check_severity(where, fk.get("severity", "error")))
        )

    return TableSpec(
        name=name,
        description=raw.get("description", ""),
        columns=columns,
        file=raw.get("file"),
        file_glob=raw.get("file_glob"),
        primary_key=primary_key,
        primary_key_severity=_check_severity(where, raw.get("primary_key_severity", "error")),
        min_rows=int(raw.get("min_rows", 1)),
        foreign_keys=tuple(foreign_keys),
    )


def parse_contract(raw: dict[str, Any]) -> SourceContract:
    source = raw["source"]
    if raw.get("format") not in FORMATS:
        raise ConfigError(f"{source}: format must be one of {sorted(FORMATS)}")
    tables = {name: _parse_table(source, name, spec) for name, spec in raw["tables"].items()}
    for table in tables.values():
        for fk in table.foreign_keys:
            for ref_table, _ in fk.references:
                if ref_table not in tables:
                    raise ConfigError(f"{source}.{table.name}: foreign key references unknown table {ref_table}")
    return SourceContract(
        source=source,
        raw_subdir=raw["raw_subdir"],
        format=raw["format"],
        tables=tables,
        layout_version=raw.get("layout_version"),
    )
