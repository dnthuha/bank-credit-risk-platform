"""Run identity and manifest: every output lives under artifacts/<run_id>/."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from credit_risk.config import ProjectConfig
from credit_risk.settings import Settings

TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "duckdb", "scipy", "scikit-learn", "pyyaml")


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    settings: Settings
    config: ProjectConfig
    started_at: str
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, status: str, detail: str = "", outputs: list[str] | None = None) -> None:
        self.steps.append({"step": step, "status": status, "detail": detail, "outputs": outputs or []})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run(settings: Settings, config: ProjectConfig) -> RunContext:
    now = _utc_now()
    base_id = f"{now:%Y%m%dT%H%M%SZ}-{config.hash[:8]}"
    run_id, suffix = base_id, 1
    while (settings.artifacts_dir / run_id).exists():
        suffix += 1
        run_id = f"{base_id}-{suffix}"
    run_dir = settings.artifacts_dir / run_id
    run_dir.mkdir(parents=True)
    return RunContext(run_id, run_dir, settings, config, now.isoformat(timespec="seconds"))


def _git_state(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_manifest(ctx: RunContext, status: str) -> Path:
    manifest = {
        "run_id": ctx.run_id,
        "status": status,
        "started_at": ctx.started_at,
        "finished_at": _utc_now().isoformat(timespec="seconds"),
        "config_hash": ctx.config.hash,
        "seeds": {
            "model_dev": ctx.config.model_dev.get("seed"),
            "validation": ctx.config.validation.get("seed"),
            "ews": ctx.config.ews.get("seed"),
        },
        "git": _git_state(ctx.settings.repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "steps": ctx.steps,
    }
    path = ctx.run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
