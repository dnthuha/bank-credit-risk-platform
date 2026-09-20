"""End-to-end pipeline: nine steps, one command, one run_id.

Steps that belong to later roadmap stages are stubs: they run, are recorded in
the manifest as "stub", and keep the chain data -> features -> model -> metrics
wired from day one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence

from credit_risk.config import ConfigError, check_definitions, load_config
from credit_risk.data.contracts import parse_contract
from credit_risk.data.db import connect, sql_ident, sql_str
from credit_risk.data.ingest import RawDataError, ingest_source, processed_path
from credit_risk.data.validate import validate_source
from credit_risk.features.split import SplitError, SplitSpec, assign_splits, split_summary
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


def _features(ctx: RunContext, sources: Sequence[str]) -> None:
    """Stage 2.1: assign the Home Credit population split, once, for every later step.

    Binning and WoE (stage 2.4) and the mortgage panel (stages 8-9) follow here.
    """
    if "home_credit" not in sources:
        ctx.record("features", "stub", "home_credit not in sources; nothing to build yet")
        return

    contract = parse_contract(ctx.config.contracts["home_credit"])
    population_path = processed_path(contract, contract.tables[ctx.config.model_dev["population"]], ctx.settings)
    if not population_path.exists():
        raise StepFailed(f"{population_path.name} not found - run ingest first")

    try:
        spec = SplitSpec.from_config(ctx.config.model_dev)
        con = connect(ctx.settings)
        columns = ", ".join(sql_ident(c) for c in [spec.id_column, spec.stratify_on] if c)
        population = con.execute(f"SELECT {columns} FROM read_parquet({sql_str(population_path)})").df()
        splits = assign_splits(population, spec)
    except SplitError as exc:
        raise StepFailed(f"split: {exc}") from exc
    out_path = population_path.with_name("splits.parquet")
    splits.to_parquet(out_path, index=False)

    summary = split_summary(splits, population, spec)
    summary_dir = ctx.run_dir / "features"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "splits_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for row in summary["splits"]:
        log.info("  split %-12s %7d rows (%.1f%%)%s", row["split"], row["rows"], 100 * row["share"],
                 f", bad rate {row['bad_rate']:.4f}" if row.get("bad_rate") is not None else "")

    ctx.record(
        "features",
        "ok",
        f"split assigned for {len(splits):,} ids (seed {spec.seed}, fingerprint {summary['fingerprint'][:12]}); "
        "binning and WoE still to come",
        [out_path.relative_to(ctx.settings.data_dir).as_posix(), "features/splits_summary.json"],
    )


def _stub(step: str, stage: str) -> Callable[[RunContext, Sequence[str]], None]:
    def run(ctx: RunContext, sources: Sequence[str]) -> None:
        log.info("[stub] %s - not implemented yet (roadmap stage %s)", step, stage)
        ctx.record(step, "stub", f"planned for roadmap stage {stage}")

    return run


STEPS: dict[str, Callable[[RunContext, Sequence[str]], None]] = {
    "ingest": _ingest,
    "validate-data": _validate_data,
    "features": _features,
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
