"""End-to-end pipeline: nine steps, one command, one run_id.

Steps that belong to later roadmap stages are stubs: they run, are recorded in
the manifest as "stub", and keep the chain data -> features -> model -> metrics
wired from day one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

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
from credit_risk.features.binning import (
    BinningError,
    BinningSpec,
    binning_frame,
    binning_payload,
    fit_all,
    load_binnings,
)
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
from credit_risk.models.calibration import (
    CalibrationError,
    CalibrationSpec,
    calibration_metrics,
    logit,
    package_fingerprint,
    reliability_table,
    select_calibrator,
    sha256_file,
    sigmoid,
)
from credit_risk.models.challenger import ChallengerError, ChallengerSpec, fit_challenger
from credit_risk.models.scorecard import Scorecard, ScorecardError, ScorecardSpec, fit_sign_stable, scorecard_payload
from credit_risk.reporting.calibration_report import anchors_table, calibration_report_markdown
from credit_risk.reporting.challenger_report import (
    agreement,
    challenger_report_markdown,
    example_rows,
    global_importance,
)
from credit_risk.reporting.model_card import SENSITIVE, discrimination, model_card_markdown, score_bands, score_psi
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


def _train(ctx: RunContext, sources: Sequence[str]) -> None:
    """Stage 3 champion scorecard, then the stage 4 LightGBM challenger, both on the Stage 2 outputs."""
    if "home_credit" not in sources:
        ctx.record("train", "stub", "home_credit not in sources; nothing to train yet")
        return

    folder = ctx.settings.processed_dir / "home_credit"
    paths = {name: folder / name for name in ("features.parquet", "splits.parquet", "binning.json", "shortlist.json")}
    for path in paths.values():
        if not path.exists():
            raise StepFailed(f"{path.name} not found - run features first")
    binning = json.loads(paths["binning.json"].read_text(encoding="utf-8"))
    shortlist = json.loads(paths["shortlist.json"].read_text(encoding="utf-8"))
    if shortlist["binning_fingerprint"] != binning["fingerprint"]:
        raise StepFailed("shortlist.json was not built on this binning.json - run features again")
    binnings = load_binnings(binning)
    out_dir = ctx.run_dir / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(ctx.settings)

    champion = _train_champion(ctx, con, paths, binning, shortlist, binnings, out_dir)
    challenger = _train_challenger(ctx, con, paths, binning, binnings, out_dir, champion)
    ctx.record("train", "ok", f"{champion['detail']}. {challenger['detail']}",
               champion["outputs"] + challenger["outputs"])


def _load_frame(con, paths, features: list[str], extra: Sequence[str] = ()) -> pd.DataFrame:
    """Applicant rows with their split (none for application_test), ordered by id."""
    available = {row[0] for row in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_str(paths['features.parquet'])})").fetchall()}
    required = ["SK_ID_CURR", "population", "TARGET", "NAME_CONTRACT_TYPE", *features]
    missing = [c for c in required if c not in available]
    if missing:
        raise StepFailed(f"features.parquet lacks {missing} - run features again")
    columns = list(dict.fromkeys([*required, *(c for c in extra if c in available)]))
    return con.execute(
        f"SELECT {', '.join('f.' + sql_ident(c) for c in columns)}, s.split "
        f"FROM read_parquet({sql_str(paths['features.parquet'])}) f "
        f"LEFT JOIN read_parquet({sql_str(paths['splits.parquet'])}) s USING (SK_ID_CURR) ORDER BY SK_ID_CURR"
    ).df()


def _train_champion(ctx: RunContext, con, paths, binning: dict, shortlist: dict, binnings, out_dir) -> dict:
    """Stage 3: the scorecard, fitted on train from the Stage 2 bins and shortlist."""
    try:
        spec = ScorecardSpec.from_config(ctx.config.model_dev, ctx.config.definitions)
    except ScorecardError as exc:
        raise StepFailed(f"scorecard: {exc}") from exc
    selected = shortlist["selected"]
    if not selected:
        raise StepFailed("the shortlist is empty: no feature to fit a scorecard on")
    folder = paths["features.parquet"].parent

    frame = _load_frame(con, paths, selected, SENSITIVE)
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    current = frame[frame["population"] == "test"]

    woe_train = pd.DataFrame({f: binnings[f].transform(train[f]) for f in selected}, index=train.index)
    try:
        fit = fit_sign_stable(woe_train, train["TARGET"].to_numpy(dtype=int), spec)
    except ScorecardError as exc:
        raise StepFailed(f"scorecard: {exc}") from exc
    card = Scorecard.build(fit, binnings, spec)

    meta = {
        "run_id": ctx.run_id,
        "fitted_on": "train",
        "rows_train": len(train),
        "split_fingerprint": binning["split_fingerprint"],
        "binning_fingerprint": binning["fingerprint"],
        "shortlist_fingerprint": shortlist["fingerprint"],
    }
    payload = scorecard_payload(card, fit, meta)
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
    stable_path = folder / "scorecard.json"
    stable_path.write_text(text, encoding="utf-8")
    (out_dir / "scorecard.json").write_text(text, encoding="utf-8")
    card.table().to_csv(out_dir / "scorecard_table.csv", index=False, encoding="utf-8")
    fit.trail.to_csv(out_dir / "sign_trail.csv", index=False, encoding="utf-8")

    # Scores and reason codes for every applicant, so stages 5-6 read one frozen output.
    points = card.points_frame(frame)
    score = card.base_points + points.sum(axis=1).to_numpy(dtype=int)
    reasons = card.reason_codes(points)
    scores = pd.DataFrame({"SK_ID_CURR": frame["SK_ID_CURR"].to_numpy(), "population": frame["population"].to_numpy(),
                           "split": frame["split"].to_numpy(), "score": score,
                           "pd_uncalibrated": card.pd_from_score(score)})
    for i in range(spec.reason_codes):
        scores[f"reason_{i + 1}"] = [r[i] if i < len(r) else None for r in reasons]
    scores_path = folder / "scores.parquet"
    scores.to_parquet(scores_path, index=False)
    score_of = pd.Series(score, index=frame.index)

    # Development performance: train and validation only.
    initial_intercept, initial_coef = fit.initial
    woe_validation = np.column_stack([binnings[f].transform(validation[f]) for f in selected])
    all_features_risk = initial_intercept + woe_validation @ np.array([initial_coef[f] for f in selected])
    auc_all_features = discrimination(validation["TARGET"], all_features_risk)["auc"]
    performance = {
        sample: {"exact": discrimination(part["TARGET"], card.exact_log_odds_bad(part)),
                 "points": discrimination(part["TARGET"], -score_of[part.index])}
        for sample, part in (("train", train), ("validation", validation))
    }
    bands = score_bands(score_of[validation.index], validation["TARGET"],
                        card.pd_from_score(score_of[validation.index]))
    stability = ctx.config.validation.get("stability") or {}
    psi_scores = score_psi(score_of[train.index], score_of[current.index], train["NAME_CONTRACT_TYPE"],
                           current["NAME_CONTRACT_TYPE"], int(stability.get("psi_n_bins", 10)),
                           float(ctx.config.definitions["metric_thresholds"]["psi"]["zero_bin_epsilon"]))
    (out_dir / "performance.json").write_text(json.dumps({
        "performance": performance,
        "auc_validation_all_features": auc_all_features,
        "score_psi_vs_application_test": psi_scores,
        "validation_bands": bands.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")

    context = {
        "meta": {**meta, "shortlisted": len(selected), "rows_validation": len(validation),
                 "bad_rate_train": float(train["TARGET"].mean()),
                 "auc_validation_all_features": auc_all_features},
        "performance": performance,
        "bands": bands,
        "psi": psi_scores,
        "iv": {name: b.iv for name, b in binnings.items()},
        "initial_coefficients": initial_coef,
        "selection_rules": shortlist["spec"],   # the Stage 2.6 rules the shortlist was actually built with
        "contract_mix": {"train": train["NAME_CONTRACT_TYPE"].value_counts(normalize=True).to_dict(),
                         "current": current["NAME_CONTRACT_TYPE"].value_counts(normalize=True).to_dict()},
    }
    (out_dir / "model_card.md").write_text(model_card_markdown(card, payload, context), encoding="utf-8")

    gini_validation = performance["validation"]["points"]["gini"]
    log.info("  scorecard: %d of %d shortlisted features kept (%d dropped for coefficient sign) | "
             "base points %d | Gini train %.4f, validation %.4f | score PSI vs application_test %.4f",
             len(card.features), len(selected), len(fit.trail), card.base_points,
             performance["train"]["points"]["gini"], gini_validation, psi_scores["all"])
    return {
        "detail": (f"scorecard: {len(card.features)} of {len(selected)} shortlisted features kept "
                   f"({len(fit.trail)} dropped for coefficient sign); validation Gini {gini_validation:.4f}; "
                   f"base points {card.base_points}; fingerprint {payload['fingerprint'][:12]}"),
        "outputs": [
            stable_path.relative_to(ctx.settings.data_dir).as_posix(),
            scores_path.relative_to(ctx.settings.data_dir).as_posix(),
            "train/scorecard.json",
            "train/scorecard_table.csv",
            "train/sign_trail.csv",
            "train/performance.json",
            "train/model_card.md",
        ],
        "performance": performance,
        "score": pd.Series(score, index=frame["SK_ID_CURR"].to_numpy()),
        "features": card.features,
        "fingerprint": payload["fingerprint"],
    }


def _train_challenger(ctx: RunContext, con, paths, binning: dict, binnings, out_dir, champion: dict) -> dict:
    """Stage 4: LightGBM on every feature of the Stage 2.3 table, tuned on validation."""
    try:
        spec = ChallengerSpec.from_config(ctx.config.model_dev)
    except ChallengerError as exc:
        raise StepFailed(f"challenger: {exc}") from exc
    folder = paths["features.parquet"].parent
    features = list(binnings)  # every feature of the table; binning.json lists them all
    categorical = [f for f in features if binnings[f].kind == "categorical"]

    frame = _load_frame(con, paths, features)
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    current = frame[frame["population"] == "test"]
    try:
        challenger = fit_challenger(train, validation, features, categorical, "TARGET", spec)
    except ChallengerError as exc:
        raise StepFailed(f"challenger: {exc}") from exc

    meta = {
        "run_id": ctx.run_id,
        "fitted_on": "train",
        "early_stopped_on": "validation",
        "rows_train": len(train),
        "rows_validation": len(validation),
        "split_fingerprint": binning["split_fingerprint"],
        "binning_fingerprint": binning["fingerprint"],
        "champion_fingerprint": champion["fingerprint"],
    }
    payload = {
        **meta,
        "spec": {**spec.__dict__},
        "fingerprint": challenger.fingerprint(),
        **challenger.to_dict(),
        "trials": challenger.trials.to_dict(orient="records"),
    }
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
    model_text = challenger.model_string()
    for directory in (folder, out_dir):
        (directory / "challenger.json").write_text(text, encoding="utf-8")
        (directory / "challenger_model.txt").write_text(model_text, encoding="utf-8")
    challenger.trials.to_csv(out_dir / "challenger_trials.csv", index=False, encoding="utf-8")

    # PD and SHAP reason codes for every applicant, in chunks to bound the contribution matrix.
    parts = []
    for start in range(0, len(frame), 50_000):
        chunk = frame.iloc[start:start + 50_000]
        contributions = challenger.contributions(chunk)
        part = pd.DataFrame({"SK_ID_CURR": chunk["SK_ID_CURR"].to_numpy(),
                             "population": chunk["population"].to_numpy(), "split": chunk["split"].to_numpy(),
                             "pd_challenger": challenger.predict_pd(chunk)})
        reasons = challenger.reason_codes(contributions)
        for i in range(spec.reason_codes):
            part[f"reason_{i + 1}"] = [r[i] if i < len(r) else None for r in reasons]
        parts.append(part)
    scores = pd.concat(parts, ignore_index=True)
    scores_path = folder / "challenger_scores.parquet"
    scores.to_parquet(scores_path, index=False)
    pd_of = pd.Series(scores["pd_challenger"].to_numpy(), index=frame.index)

    performance = {sample: discrimination(part["TARGET"], pd_of[part.index])
                   for sample, part in (("train", train), ("validation", validation))}
    champion_perf = {sample: champion["performance"][sample]["points"] for sample in ("train", "validation")}
    agree = agreement(pd_of[validation.index], champion["score"].loc[validation["SK_ID_CURR"]].to_numpy(),
                      validation["TARGET"])
    contributions = challenger.contributions(validation)
    importance = global_importance(contributions, challenger)
    importance.to_csv(out_dir / "challenger_importance.csv", index=False, encoding="utf-8")
    stability = ctx.config.validation.get("stability") or {}
    psi_pd = score_psi(pd_of[train.index], pd_of[current.index], train["NAME_CONTRACT_TYPE"],
                       current["NAME_CONTRACT_TYPE"], int(stability.get("psi_n_bins", 10)),
                       float(ctx.config.definitions["metric_thresholds"]["psi"]["zero_bin_epsilon"]))

    examples = []
    for idx in example_rows(pd_of[validation.index], spec.shap_examples):
        row = contributions.loc[idx, challenger.features]
        top = row.reindex(row.abs().sort_values(ascending=False, kind="stable").index)[:6]
        examples.append({
            "id": int(frame.loc[idx, "SK_ID_CURR"]), "pd": float(pd_of[idx]), "target": int(frame.loc[idx, "TARGET"]),
            "base": float(contributions.loc[idx, "base_value"]),
            "total": float(contributions.loc[idx, "base_value"] + row.sum()),
            "top": [(name, frame.loc[idx, name], float(value)) for name, value in top.items()],
            "rest": float(row.sum() - top.sum()), "n_rest": len(row) - len(top),
        })
    (out_dir / "challenger_performance.json").write_text(json.dumps({
        "performance": performance, "champion": champion_perf, "agreement": agree,
        "pd_psi_vs_application_test": psi_pd, "examples": examples,
    }, indent=2, default=str), encoding="utf-8")

    context = {
        "meta": {**meta, "champion_features": champion["features"]},
        "performance": performance,
        "champion": champion_perf,
        "agreement": agree,
        "importance": importance,
        "psi": psi_pd,
        "examples": examples,
    }
    (out_dir / "challenger_report.md").write_text(challenger_report_markdown(challenger, payload, context),
                                                  encoding="utf-8")

    gini_validation = performance["validation"]["gini"]
    log.info("  challenger: best of %d trials, %d trees | Gini train %.4f, validation %.4f "
             "(champion %.4f) | PD PSI vs application_test %.4f",
             len(challenger.trials), challenger.best_iteration, performance["train"]["gini"], gini_validation,
             champion_perf["validation"]["gini"], psi_pd["all"])
    return {
        "detail": (f"challenger: LightGBM on {len(features)} features, best of {len(challenger.trials)} trials, "
                   f"{challenger.best_iteration} trees; validation Gini {gini_validation:.4f} "
                   f"({gini_validation - champion_perf['validation']['gini']:+.4f} vs champion); "
                   f"fingerprint {payload['fingerprint'][:12]}"),
        "outputs": [
            (folder / "challenger.json").relative_to(ctx.settings.data_dir).as_posix(),
            (folder / "challenger_model.txt").relative_to(ctx.settings.data_dir).as_posix(),
            scores_path.relative_to(ctx.settings.data_dir).as_posix(),
            "train/challenger.json",
            "train/challenger_model.txt",
            "train/challenger_trials.csv",
            "train/challenger_importance.csv",
            "train/challenger_performance.json",
            "train/challenger_report.md",
        ],
    }


PACKAGE_DIR = "model_package"


def _display_path(path, settings: Settings) -> str:
    """Path as a reader finds it: relative to the repository when it lives inside it."""
    try:
        return path.relative_to(settings.repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _calibrate(ctx: RunContext, sources: Sequence[str]) -> None:
    """Stage 5: calibrate both models' PD on the calibration split, then freeze the model package."""
    if "home_credit" not in sources:
        ctx.record("calibrate", "stub", "home_credit not in sources; nothing to calibrate yet")
        return

    folder = ctx.settings.processed_dir / "home_credit"
    names = ("scorecard.json", "scores.parquet", "challenger.json", "challenger_model.txt",
             "challenger_scores.parquet", "features.parquet", "splits.parquet")
    paths = {name: folder / name for name in names}
    for path in paths.values():
        if not path.exists():
            raise StepFailed(f"{path.name} not found - run train first")
    try:
        spec = CalibrationSpec.from_config(ctx.config.model_dev)
    except CalibrationError as exc:
        raise StepFailed(f"calibration: {exc}") from exc
    scorecard = json.loads(paths["scorecard.json"].read_text(encoding="utf-8"))
    challenger = json.loads(paths["challenger.json"].read_text(encoding="utf-8"))
    if challenger["champion_fingerprint"] != scorecard["fingerprint"]:
        raise StepFailed("challenger.json was trained next to another scorecard - run train again")

    con = connect(ctx.settings)
    n_reasons = sum(1 for c in pd.read_parquet(paths["scores.parquet"]).columns if c.startswith("reason_"))
    reasons = [f"reason_{i}" for i in range(1, n_reasons + 1)]
    champion_cols = ", ".join(f"s.{c} AS champion_{c}" for c in reasons)
    challenger_cols = ", ".join(f"c.{c} AS challenger_{c}" for c in reasons)
    frame = con.execute(
        f"SELECT s.SK_ID_CURR, s.population, s.split, s.score, c.pd_challenger, {champion_cols}, {challenger_cols}, "
        "f.TARGET, f.NAME_CONTRACT_TYPE "
        f"FROM read_parquet({sql_str(paths['scores.parquet'])}) s "
        f"JOIN read_parquet({sql_str(paths['challenger_scores.parquet'])}) c USING (SK_ID_CURR) "
        f"JOIN read_parquet({sql_str(paths['features.parquet'])}) f USING (SK_ID_CURR) ORDER BY SK_ID_CURR"
    ).df()
    counts = [len(pd.read_parquet(paths[n], columns=["SK_ID_CURR"]))
              for n in ("scores.parquet", "challenger_scores.parquet", "features.parquet")]
    if len(set(counts + [len(frame)])) != 1:
        raise StepFailed("scores.parquet, challenger_scores.parquet and features.parquet do not cover the same "
                         "applicants - run train again")

    offset, factor = float(scorecard["offset"]), float(scorecard["factor"])
    log_odds = {"champion": -(frame["score"].to_numpy(dtype=float) - offset) / factor,
                "challenger": logit(np.clip(frame["pd_challenger"].to_numpy(dtype=float), 1e-15, 1 - 1e-15))}
    masks = {s: (frame["split"] == s).to_numpy() for s in ("train", "validation", "calibration")}
    y = frame["TARGET"].to_numpy()

    models, calibrators, calibrated = {}, {}, {}
    for name, x in log_odds.items():
        cal = masks["calibration"]
        try:
            selection = select_calibrator(x[cal], y[cal].astype(int), spec)
        except CalibrationError as exc:
            raise StepFailed(f"calibration ({name}): {exc}") from exc
        calibrator = selection.calibrator
        pd_after = calibrator.apply(x)
        pd_before = sigmoid(x)
        va = masks["validation"]
        by_contract = []
        for contract, part in frame[va].assign(pd=pd_after[va]).groupby("NAME_CONTRACT_TYPE"):
            by_contract.append({"segment": contract, "n": len(part), "mean_pd": float(part["pd"].mean()),
                                "bad_rate": float(part["TARGET"].mean()),
                                "ratio": float(part["TARGET"].mean() / part["pd"].mean())})
        models[name] = {
            "before": {s: calibration_metrics(y[m], pd_before[m]) for s, m in masks.items()},
            "cv": selection.cv.to_dict(orient="records"),
            "chosen": calibrator.method,
            "params": calibrator.params,
            "after": {"validation": calibration_metrics(y[va], pd_after[va])},
            "auc": {"before": discrimination(y[va], pd_before[va])["auc"],
                    "after": discrimination(y[va], pd_after[va])["auc"]},
            "reliability": reliability_table(y[va], pd_after[va], spec.n_bins).to_dict(orient="records"),
            "by_contract": by_contract,
        }
        calibrators[name], calibrated[name] = calibrator, pd_after

    # Freeze: the two models, their calibrators and the calibrated outputs, in one folder.
    package_dir = folder / PACKAGE_DIR
    package_dir.mkdir(exist_ok=True)
    (package_dir / "champion.json").write_text(json.dumps(
        {"scorecard": scorecard, "calibration": calibrators["champion"].to_dict()},
        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (package_dir / "challenger.json").write_text(json.dumps(
        {"challenger": challenger, "calibration": calibrators["challenger"].to_dict()},
        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (package_dir / "challenger_model.txt").write_bytes(paths["challenger_model.txt"].read_bytes())
    out = pd.DataFrame({"SK_ID_CURR": frame["SK_ID_CURR"], "population": frame["population"],
                        "split": frame["split"], "score": frame["score"],
                        "pd_champion": calibrated["champion"], "pd_challenger": calibrated["challenger"]})
    for column in reasons:
        out[f"champion_{column}"] = frame[f"champion_{column}"]
        out[f"challenger_{column}"] = frame[f"challenger_{column}"]
    out.to_parquet(package_dir / "calibrated_scores.parquet", index=False)

    content = {"champion": scorecard["fingerprint"], "challenger": challenger["fingerprint"],
               "calibration": {n: c.to_dict() for n, c in calibrators.items()}}
    files = {name: sha256_file(package_dir / name)
             for name in ("champion.json", "challenger.json", "challenger_model.txt", "calibrated_scores.parquet")}
    package = {
        "run_id": ctx.run_id,
        "config_hash": ctx.config.hash,
        "frozen_at": ctx.started_at,
        "fitted_on": "calibration",
        "split_fingerprint": scorecard["split_fingerprint"],
        "champion_fingerprint": scorecard["fingerprint"],
        "challenger_fingerprint": challenger["fingerprint"],
        "calibration": {n: c.method for n, c in calibrators.items()},
        "fingerprint": package_fingerprint(content),
        "files": files,
    }
    (package_dir / "package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")

    out_dir = ctx.run_dir / "calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    (out_dir / "calibration.json").write_text(json.dumps(models, indent=2, default=str), encoding="utf-8")
    context = {
        "meta": {"run_id": ctx.run_id, "champion_fingerprint": scorecard["fingerprint"],
                 "challenger_fingerprint": challenger["fingerprint"],
                 "rows_calibration": int(masks["calibration"].sum()), "rows_validation": int(masks["validation"].sum()),
                 "bad_rate_calibration": float(y[masks["calibration"]].mean())},
        "spec": spec.__dict__,
        "models": models,
        "anchors": anchors_table(offset, factor, calibrators["champion"]),
        "package": {**package, "folder": _display_path(package_dir, ctx.settings)},
    }
    (out_dir / "calibration_report.md").write_text(calibration_report_markdown(context), encoding="utf-8")

    summary = "; ".join(
        f"{n}: {m['chosen']} (validation Brier {m['before']['validation']['brier']:.5f} -> "
        f"{m['after']['validation']['brier']:.5f}, slope {m['after']['validation']['slope']:.3f}, "
        f"intercept {m['after']['validation']['intercept']:+.3f})" for n, m in models.items())
    log.info("  calibration: %s | package %s", summary, package["fingerprint"][:12])
    ctx.record("calibrate", "ok", f"{summary}; package fingerprint {package['fingerprint'][:12]}", [
        *(f"{package_dir.relative_to(ctx.settings.data_dir).as_posix()}/{n}" for n in [*files, "package.json"]),
        "calibrate/package.json",
        "calibrate/calibration.json",
        "calibrate/calibration_report.md",
    ])


def _validate_model(ctx: RunContext, sources: Sequence[str]) -> None:
    """Stages 6-7: independent validation of the frozen package - recount everything, conclude, write metrics + report."""
    if "home_credit" not in sources:
        ctx.record("validate-model", "stub", "home_credit not in sources; nothing to validate yet")
        return
    from credit_risk.validation.engine import ValidationFailed, run_validation, write_outputs

    out_dir = ctx.run_dir / "validate-model"
    try:
        result = run_validation(ctx.settings, ctx.config, ctx.run_id)
    except ValidationFailed as exc:
        raise StepFailed(f"validate-model: {exc}") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise StepFailed(f"validate-model: {exc}") from exc
    paths = write_outputs(result, out_dir, ctx.settings)
    p = result.provenance
    summary = (f"package {p['package_fingerprint'][:12]}: {p['checks_total']} checks, {p['checks_failed']} failed, "
               f"{p['findings']} findings; champion {p['verdicts']['champion']}, challenger {p['verdicts']['challenger']}, "
               f"deploy {p['deploy'] or 'none'}")
    log.info("  validate-model: %s", summary)
    ctx.record("validate-model", "ok", summary, [f"validate-model/{path.name}" for path in paths])


def _monitor(ctx: RunContext, sources: Sequence[str]) -> None:
    """Stage 8: portfolio monitoring on the Freddie Mac panel - metrics, curves, migration, concentration, alerts."""
    if "freddie_mac" not in sources:
        ctx.record("monitor", "stub", "freddie_mac not in sources; nothing to monitor")
        return
    from credit_risk.monitoring.engine import MonitoringFailed, run_monitoring, write_outputs

    try:
        result = run_monitoring(ctx.settings, ctx.config, ctx.run_id)
    except (MonitoringFailed, FileNotFoundError, KeyError, ValueError) as exc:
        raise StepFailed(f"monitor: {exc}") from exc
    paths = write_outputs(result, ctx.run_dir / "monitor")
    p = result.provenance
    row = result.tables["portfolio_monthly"].set_index("period").loc[result.month]
    detail = (f"{p['reporting_month']}: status {result.status} ({p['alerts']['red']} red, {p['alerts']['amber']} amber); "
              f"90+ {row['dpd90plus_rate_count']:.3%} of loans / {row['dpd90plus_rate_upb']:.3%} of UPB; "
              f"panel {p['panel']['rows']:,} rows, {p['panel']['loans']:,} loans; "
              f"{p['checks_total']} checks, {p['checks_failed']} failed")
    log.info("  monitor: %s", detail)
    ctx.record("monitor", "ok", detail, [result.panel_path.relative_to(ctx.settings.data_dir).as_posix(),
                                         *(f"monitor/{path.name}" for path in paths)])



def _ews(ctx: RunContext, sources: Sequence[str]) -> None:
    """Stage 9: early warning on the Freddie Mac panel - label, features, models, watchlist, survival."""
    if "freddie_mac" not in sources:
        ctx.record("ews", "stub", "freddie_mac not in sources; nothing to score")
        return
    from credit_risk.ews.engine import EwsFailed, run_ews, write_outputs

    try:
        result = run_ews(ctx.settings, ctx.config, ctx.run_id)
    except (EwsFailed, FileNotFoundError, KeyError, ValueError) as exc:
        raise StepFailed(f"ews: {exc}") from exc
    paths = write_outputs(result, ctx.run_dir / "ews")
    p = result.provenance
    oot = p["out_of_time"]
    detail = (f"selected {p['selected_model']}; out-of-time "
              + (f"PR AUC {oot.get('pr_auc', float('nan')):.3f}, capture top 5% {oot.get('capture_top_5pct', float('nan')):.1%}; "
                 if oot else "not available; ")
              + f"watchlist {p['scoring_month']}: {p['watchlist']['listed']:,} of {p['watchlist']['scoring_rows']:,} loans; "
              f"{p['checks_total']} checks, {p['checks_failed']} failed")
    log.info("  ews: %s", detail)
    ctx.record("ews", "ok", detail, [*(f"ews/{path.relative_to(ctx.run_dir / 'ews').as_posix()}" for path in paths)])

def _stub(step: str, stage: str) -> Callable[[RunContext, Sequence[str]], None]:
    def run(ctx: RunContext, sources: Sequence[str]) -> None:
        log.info("[stub] %s - not implemented yet (roadmap stage %s)", step, stage)
        ctx.record(step, "stub", f"planned for roadmap stage {stage}")

    return run


STEPS: dict[str, Callable[[RunContext, Sequence[str]], None]] = {
    "ingest": _ingest,
    "validate-data": _validate_data,
    "features": _features,
    "train": _train,
    "calibrate": _calibrate,
    "validate-model": _validate_model,
    "monitor": _monitor,
    "ews": _ews,
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
