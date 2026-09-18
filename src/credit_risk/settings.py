"""Machine-specific runtime settings, read from environment variables or .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    config_dir: Path
    data_dir: Path
    artifacts_dir: Path
    duckdb_memory_limit: str
    duckdb_threads: int

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_settings(repo_root: Path = REPO_ROOT) -> Settings:
    # Real environment variables win over .env, so tests and CI can override.
    load_dotenv(repo_root / ".env", override=False)
    env = os.environ.get
    return Settings(
        repo_root=repo_root,
        config_dir=_resolve(env("CREDIT_RISK_CONFIG_DIR", "configs"), repo_root),
        data_dir=_resolve(env("CREDIT_RISK_DATA_DIR", "data"), repo_root),
        artifacts_dir=_resolve(env("CREDIT_RISK_ARTIFACTS_DIR", "artifacts"), repo_root),
        duckdb_memory_limit=env("CREDIT_RISK_DUCKDB_MEMORY_LIMIT", "3GB"),
        duckdb_threads=int(env("CREDIT_RISK_DUCKDB_THREADS", "4")),
    )
