"""DuckDB connection tuned for a low-RAM laptop: bounded memory, spill to disk."""

from __future__ import annotations

from pathlib import Path

import duckdb

from credit_risk.settings import Settings


def sql_str(value: str | Path) -> str:
    """Quote a value as a SQL string literal (paths use forward slashes)."""
    text = value.as_posix() if isinstance(value, Path) else str(value)
    return "'" + text.replace("'", "''") + "'"


def sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    tmp_dir = settings.data_dir / ".duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = {sql_str(settings.duckdb_memory_limit)}")
    con.execute(f"SET threads = {int(settings.duckdb_threads)}")
    con.execute(f"SET temp_directory = {sql_str(tmp_dir)}")
    con.execute("SET preserve_insertion_order = false")
    return con
