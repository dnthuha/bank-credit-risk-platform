"""End-to-end pipeline: nine steps, one command, one run_id.

Steps that belong to later roadmap stages are stubs: they run, are recorded in
the manifest as "stub", and keep the chain data -> features -> model -> metrics
wired from day one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from credit_risk.config import ConfigError, check_definitions, load_config
from credit_risk.data.contracts import parse_contract
from credit_risk.data.db import connect
from credit_risk.data.ingest import RawDataError, ingest_source
from credit_risk.data.validate import validate_source
from credit_risk.lineage import RunContext, new_run, write_manifest
from credit_risk.settings import Settings, load_settings

log = logging.getLogger(__name__)

SOURCES = ("home_credit", "freddie_mac")


class StepFailed(RuntimeError):
    pass


def _ingest(ctx: RunContext, sources: Sequence[str]) -> None:
    con = connect(ctx.settings)
    outputs = []
    for source in sources:
        contract = parse_contract(ctx.config.contracts[source])
        try:
            manifest = ingest_source(contract, ctx.settings, con, ctx.run_id)
        except RawDataError as exc:
            raise StepFailed(str(exc)) from exc
        outputs += [t["output"] for t in manifest["tables"].values()]
    ctx.record("ingest", "ok", f"sources={list(sources)}", outputs)


def _validate_data(ctx: RunContext, sources: Sequence[str]) -> None:
    con = connect(ctx.settings)
    out_dir = ctx.run_dir / "data_validation"
    failed, outputs, summary = [], [], []
    for source in sources:
        report = validate_source(parse_contract(ctx.config.contracts[source]), ctx.settings, con)
        outputs += [p.relative_to(ctx.run_dir).as_posix() for p in report.write(out_dir)]
        summary.append(f"{source}: {len(report.errors)} errors, {len(report.warnings)} warnings")
        for r in report.errors:
            log.error("[%s] %s.%s %s: %s", source, r.table, r.column or "", r.check, r.detail)
        for r in report.warnings:
            log.warning("[%s] %s.%s %s: %s", source, r.table, r.column or "", r.check, r.detail)
        if not report.passed:
            failed.append(source)
    if failed:
        ctx.record("validate-data", "failed", "; ".join(summary), outputs)
        raise StepFailed(f"data validation failed for {failed}; see {out_dir}")
    ctx.record("validate-data", "ok", "; ".join(summary), outputs)


def _stub(step: str, stage: str) -> Callable[[RunContext, Sequence[str]], None]:
    def run(ctx: RunContext, sources: Sequence[str]) -> None:
        log.info("[stub] %s - not implemented yet (roadmap stage %s)", step, stage)
        ctx.record(step, "stub", f"planned for roadmap stage {stage}")

    return run


STEPS: dict[str, Callable[[RunContext, Sequence[str]], None]] = {
    "ingest": _ingest,
    "validate-data": _validate_data,
    "features": _stub("features", "2 (WoE/IV) and 8-9 (panel)"),
    "train": _stub("train", "3-4"),
    "calibrate": _stub("calibrate", "5"),
    "validate-model": _stub("validate-model", "6"),
    "monitor": _stub("monitor", "8"),
    "ews": _stub("ews", "9"),
    "report": _stub("report", "7 and 10"),
}


def run_pipeline(
    steps: Sequence[str] | None = None,
    sources: Sequence[str] = SOURCES,
    settings: Settings | None = None,
) -> int:
    """Run the given steps in canonical order. Returns a process exit code."""
    settings = settings or load_settings()
    requested = list(STEPS) if not steps else [s for s in STEPS if s in set(steps)]
    unknown = set(steps or []) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps: {sorted(unknown)}")
    unknown_sources = set(sources) - set(SOURCES)
    if unknown_sources:
        raise ValueError(f"unknown sources: {sorted(unknown_sources)}")

    config = load_config(settings.config_dir)
    problems = check_definitions(config.definitions)
    if problems:
        raise ConfigError("definitions.yaml is incoherent: " + "; ".join(problems))

    ctx = new_run(settings, config)
    log.info("run %s | steps=%s | sources=%s", ctx.run_id, requested, list(sources))
    status = "ok"
    for step in requested:
        try:
            STEPS[step](ctx, sources)
        except StepFailed as exc:
            log.error("step %s failed: %s", step, exc)
            if not any(s["step"] == step for s in ctx.steps):
                ctx.record(step, "failed", str(exc))
            status = "failed"
            break
    manifest = write_manifest(ctx, status)
    log.info("run %s finished: %s (manifest: %s)", ctx.run_id, status, manifest)
    return 0 if status == "ok" else 1
