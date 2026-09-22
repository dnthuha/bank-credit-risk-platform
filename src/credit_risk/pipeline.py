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
from credit_risk.features.availability import (
    AvailabilityError,
    availability_frame,
    availability_markdown,
    parse_availability,
)
from credit_risk.features.binning import BinningError, BinningSpec, binning_frame, binning_payload, fit_all
from credit_risk.features.build import build_features
from credit_risk.features.iv_report import ReportThresholds, iv_report, iv_report_markdown
from credit_risk.features.selection import (
    SelectionError,
    SelectionSpec,
    select_features,
    shortlist_markdown,
    shortlist_payload,
)
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
    """Stage 2 for Home Credit, in order: population split (2.1), availability
    matrix (2.2), applicant-level feature table (2.3), binning fitted on train
    (2.4), out-of-sample IV and stability report (2.5), feature shortlist (2.6).

    The mortgage panel (stages 8-9) will join this step later.
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
    matrix, n_classified = _write_availability_matrix(ctx, contract, con, summary_dir)

    features = build_features(contract, matrix, ctx.settings, con)
    (summary_dir / "features_summary.json").write_text(
        json.dumps(features, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    binnings, binning = _fit_binning(ctx, con, features, out_path, summary["fingerprint"], summary_dir)
    report = _write_iv_report(ctx, con, features, out_path, binnings, binning, summary_dir)
    shortlist = _select_features(ctx, con, features, out_path, binnings, binning, report, summary_dir)
    for row in summary["splits"]:
        log.info("  split %-12s %7d rows (%.1f%%)%s", row["split"], row["rows"], 100 * row["share"],
                 f", bad rate {row['bad_rate']:.4f}" if row.get("bad_rate") is not None else "")

    ctx.record(
        "features",
        "ok",
        f"split assigned for {len(splits):,} ids (seed {spec.seed}, fingerprint {summary['fingerprint'][:12]}); "
        f"{n_classified} usable columns in the availability matrix; "
        f"{features['columns']} column feature table for {features['rows']:,} applicants; "
        f"{binning['features']} features binned on {binning['rows']:,} train rows "
        f"({binning['flagged_for_review']} flagged for review, fingerprint {binning['fingerprint'][:12]}); "
        f"{report['useful']} features with IV >= {report['iv_min']} on train, {report['flagged']} of them flagged "
        f"out of sample; shortlist of {shortlist['selected']} features (fingerprint {shortlist['fingerprint'][:12]})",
        [
            out_path.relative_to(ctx.settings.data_dir).as_posix(),
            features["output"],
            binning["output"],
            "features/splits_summary.json",
            "features/features_summary.json",
            "features/feature_availability.md",
            "features/feature_availability.csv",
            "features/binning.json",
            "features/binning_table.csv",
            "features/iv_report.md",
            "features/iv_report.csv",
            "features/bin_stability.csv",
            shortlist["output"],
            "features/shortlist.md",
            "features/shortlist.csv",
            "features/shortlist.json",
        ],
    )


def _fit_binning(ctx: RunContext, con, features: dict, splits_path, split_fingerprint: str, out_dir):
    """Stage 2.4: fit every bin on the train split only, then persist the table for scoring."""
    try:
        spec = BinningSpec.from_config(ctx.config.model_dev, ctx.config.definitions)
    except BinningError as exc:
        raise StepFailed(f"binning: {exc}") from exc

    features_path = ctx.settings.data_dir / features["output"]
    train = con.execute(
        f"SELECT f.* FROM read_parquet({sql_str(features_path)}) f "
        f"JOIN read_parquet({sql_str(splits_path)}) s USING (SK_ID_CURR) WHERE s.split = 'train'"
    ).df()
    try:
        binnings = fit_all(train, features["feature_columns"], "TARGET", spec)
    except BinningError as exc:
        raise StepFailed(f"binning: {exc}") from exc

    payload = binning_payload(binnings, spec, {
        "run_id": ctx.run_id,
        "fitted_on": "train",
        "rows": len(train),
        "split_fingerprint": split_fingerprint,
    })
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
    stable_path = features_path.with_name("binning.json")
    stable_path.write_text(text, encoding="utf-8")
    (out_dir / "binning.json").write_text(text, encoding="utf-8")
    binning_frame(binnings).to_csv(out_dir / "binning_table.csv", index=False, encoding="utf-8")

    flagged = [name for name, b in binnings.items() if any("review" in note for note in b.notes)]
    suspicious = [name for name, b in binnings.items()
                  if b.iv > ctx.config.definitions["metric_thresholds"]["iv"]["leakage_suspect_above"]]
    log.info("  binning: %d features on %d train rows | %d flagged for review | IV above leakage threshold: %s",
             len(binnings), len(train), len(flagged), suspicious or "none")
    return binnings, {
        "features": len(binnings),
        "rows": len(train),
        "flagged_for_review": len(flagged),
        "fingerprint": payload["fingerprint"],
        "output": stable_path.relative_to(ctx.settings.data_dir).as_posix(),
    }


def _write_iv_report(ctx: RunContext, con, features: dict, splits_path, binnings, binning: dict, out_dir) -> dict:
    """Stage 2.5: apply the train bins to validation and to application_test; flag, never refit."""
    features_path = ctx.settings.data_dir / features["output"]
    base = f"read_parquet({sql_str(features_path)})"
    validation = con.execute(
        f"SELECT f.* FROM {base} f JOIN read_parquet({sql_str(splits_path)}) s USING (SK_ID_CURR) "
        "WHERE s.split = 'validation'"
    ).df()
    current = con.execute(f"SELECT * FROM {base} WHERE population = 'test'").df()

    th = ReportThresholds.from_config(ctx.config.definitions, ctx.config.model_dev)
    table, bins = iv_report(binnings, validation, current, "TARGET", th)
    table.to_csv(out_dir / "iv_report.csv", index=False, encoding="utf-8")
    bins.to_csv(out_dir / "bin_stability.csv", index=False, encoding="utf-8")
    meta = {"rows_train": binning["rows"], "rows_validation": len(validation), "rows_current": len(current)}
    (out_dir / "iv_report.md").write_text(iv_report_markdown(table, th, meta), encoding="utf-8")

    useful = table[table["iv_train"] >= th.iv_min]
    out_of_sample = {"iv_drops_out_of_sample", "trend_not_confirmed", "unstable_vs_validation"}
    flagged = useful[useful["review_flags"].map(lambda f: bool(out_of_sample & set(f.split(";"))))]
    log.info("  iv report: %d useful features on train | %d flagged out of sample | "
             "median IV kept on validation %.0f%% | PSI vs application_test >= %.2f: %d features",
             len(useful), len(flagged), 100 * useful["iv_retention"].median(),
             th.psi_significant, int((table["psi_current"] >= th.psi_significant).sum()))
    return {"useful": len(useful), "flagged": len(flagged), "iv_min": th.iv_min, "table": table}


def _select_features(ctx: RunContext, con, features: dict, splits_path, binnings, binning: dict,
                     report: dict, out_dir) -> dict:
    """Stage 2.6: the shortlist, with the step and reason that removed every other feature."""
    try:
        spec = SelectionSpec.from_config(ctx.config.model_dev, ctx.config.definitions)
    except SelectionError as exc:
        raise StepFailed(f"selection: {exc}") from exc

    features_path = ctx.settings.data_dir / features["output"]
    train = con.execute(
        f"SELECT f.* FROM read_parquet({sql_str(features_path)}) f "
        f"JOIN read_parquet({sql_str(splits_path)}) s USING (SK_ID_CURR) WHERE s.split = 'train'"
    ).df()
    decisions = select_features(report["table"], binnings, train, spec)

    meta = {"run_id": ctx.run_id, "rows_train": len(train), "binning_fingerprint": binning["fingerprint"]}
    payload = shortlist_payload(decisions, spec, meta)
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
    stable_path = features_path.with_name("shortlist.json")
    stable_path.write_text(text, encoding="utf-8")
    (out_dir / "shortlist.json").write_text(text, encoding="utf-8")
    decisions.to_csv(out_dir / "shortlist.csv", index=False, encoding="utf-8")
    (out_dir / "shortlist.md").write_text(shortlist_markdown(decisions, spec, meta), encoding="utf-8")

    counts = decisions["step"].value_counts().to_dict()
    log.info("  shortlist: %d selected | dropped by %s", counts.get("selected", 0),
             ", ".join(f"{k} {v}" for k, v in counts.items() if k != "selected"))
    return {
        "selected": counts.get("selected", 0),
        "fingerprint": payload["fingerprint"],
        "output": stable_path.relative_to(ctx.settings.data_dir).as_posix(),
    }


def _write_availability_matrix(ctx: RunContext, contract, con, out_dir):
    """Classify every column of every ingested table; an unclassified column stops the step."""
    columns_by_table = {}
    for name, table in contract.tables.items():
        path = processed_path(contract, table, ctx.settings)
        if not path.exists():
            raise StepFailed(f"{path.name} not found - run ingest first")
        described = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_str(path)})").fetchall()
        columns_by_table[name] = [row[0] for row in described]

    try:
        matrix = parse_availability(ctx.config.feature_availability, contract.source)
        frame = availability_frame(matrix, columns_by_table)
    except AvailabilityError as exc:
        raise StepFailed(f"feature availability: {exc}") from exc

    frame.to_csv(out_dir / "feature_availability.csv", index=False, encoding="utf-8")
    (out_dir / "feature_availability.md").write_text(
        availability_markdown(matrix, frame), encoding="utf-8"
    )
    counts = frame["availability"].value_counts().to_dict()
    log.info("  availability: %s | usable features: %d of %d columns",
             ", ".join(f"{k} {v}" for k, v in counts.items()), int(frame["use"].sum()), len(frame))
    return matrix, int(frame["use"].sum())


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
