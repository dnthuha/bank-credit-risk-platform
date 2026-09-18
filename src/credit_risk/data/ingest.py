"""Step 1 - ingest: raw files -> typed parquet under data/processed/<source>/.

Raw files are only read, never modified. Type standardisation happens here;
business transformations do not.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from credit_risk.data.contracts import ColumnSpec, SourceContract, TableSpec
from credit_risk.data.db import sql_ident, sql_str
from credit_risk.settings import Settings

log = logging.getLogger(__name__)


class RawDataError(RuntimeError):
    pass


def raw_files(contract: SourceContract, table: TableSpec, settings: Settings) -> list[Path]:
    raw_dir = settings.raw_dir / contract.raw_subdir
    if table.file:
        path = raw_dir / table.file
        return [path] if path.exists() else []
    return sorted(raw_dir.glob(table.file_glob))


def processed_path(contract: SourceContract, table: TableSpec, settings: Settings) -> Path:
    return settings.processed_dir / contract.source / f"{table.name}.parquet"


def _check_raw_present(contract: SourceContract, settings: Settings) -> dict[str, list[Path]]:
    found = {name: raw_files(contract, table, settings) for name, table in contract.tables.items()}
    missing = [
        contract.tables[name].file or contract.tables[name].file_glob
        for name, files in found.items()
        if not files
    ]
    if missing:
        raise RawDataError(
            f"[{contract.source}] raw files not found in "
            f"{settings.raw_dir / contract.raw_subdir}: {missing}. See data/README.md."
        )
    return found


def _check_field_count(contract: SourceContract, table: TableSpec, files: list[Path]) -> None:
    expected = len(table.columns)
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as fh:
            first = fh.readline().rstrip("\r\n")
        if not first:
            raise RawDataError(f"[{contract.source}] empty file: {path.name}")
        actual = len(first.split("|"))
        if actual != expected:
            raise RawDataError(
                f"[{contract.source}] {path.name} has {actual} fields, contract "
                f"(layout {contract.layout_version}) expects {expected}. The file layout "
                f"probably changed: update configs/data_contracts/{contract.source}.yaml."
            )


def _typed_expr(col: ColumnSpec, raw_name: str) -> str:
    raw = f"trim({sql_ident(raw_name)})"
    null_tokens = ["''", *(sql_str(v) for v in col.na_values)]
    value = f"CASE WHEN {raw} IN ({', '.join(null_tokens)}) THEN NULL ELSE {raw} END"
    # CAST (not TRY_CAST): a value that does not fit the contract type must fail loudly.
    if col.dtype == "integer":
        value = f"CAST({value} AS BIGINT)"
    elif col.dtype == "number":
        value = f"CAST({value} AS DOUBLE)"
    elif col.dtype == "yyyymm":
        value = f"CAST(strptime({value}, '%Y%m') AS DATE)"
    elif col.dtype == "date":
        value = f"CAST({value} AS DATE)"
    return f"{value} AS {sql_ident(col.name)}"


def _select_sql(contract: SourceContract, table: TableSpec, files: list[Path]) -> str:
    file_list = "[" + ", ".join(sql_str(p) for p in files) + "]"
    if contract.format == "csv":
        return f"SELECT * FROM read_csv({file_list}, header = true, sample_size = -1)"

    # pipe_no_header: read everything as text in contract order, then cast.
    raw_names = [f"c{i:02d}" for i in range(len(table.columns))]
    columns_struct = "{" + ", ".join(f"{sql_str(n)}: 'VARCHAR'" for n in raw_names) + "}"
    typed = ",\n  ".join(_typed_expr(col, raw) for col, raw in zip(table.columns, raw_names))
    return (
        f"SELECT\n  {typed},\n  parse_filename(filename) AS source_file\n"
        f"FROM read_csv({file_list}, delim = '|', header = false, quote = '', escape = '', "
        f"columns = {columns_struct}, filename = true)"
    )


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }


def ingest_source(
    contract: SourceContract, settings: Settings, con: duckdb.DuckDBPyConnection, run_id: str
) -> dict[str, Any]:
    found = _check_raw_present(contract, settings)
    out_dir = settings.processed_dir / contract.source
    out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, Any] = {}
    for name, table in contract.tables.items():
        files = found[name]
        if contract.format == "pipe_no_header":
            _check_field_count(contract, table, files)
        out_path = processed_path(contract, table, settings)
        tmp_path = out_path.with_suffix(".parquet.tmp")
        log.info("ingest %s.%s from %d file(s)", contract.source, name, len(files))
        con.execute(
            f"COPY ({_select_sql(contract, table, files)}) TO {sql_str(tmp_path)} "
            f"(FORMAT parquet, COMPRESSION zstd)"
        )
        tmp_path.replace(out_path)  # only replace the previous table once the write succeeded
        n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_str(out_path)})").fetchone()[0]
        tables[name] = {
            "rows": n_rows,
            "output": out_path.relative_to(settings.data_dir).as_posix(),
            "raw_files": [_file_fingerprint(p) for p in files],
        }
        log.info("  -> %s rows", f"{n_rows:,}")

    manifest = {
        "source": contract.source,
        "layout_version": contract.layout_version,
        "run_id": run_id,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tables": tables,
    }
    (out_dir / "_ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
