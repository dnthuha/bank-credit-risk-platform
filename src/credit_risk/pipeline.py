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
from credit_risk.models.challenger import ChallengerError, ChallengerSpec, fit_challenger
from credit_risk.models.scorecard import Scorecard, ScorecardError, ScorecardSpec, fit_sign_stable, scorecard_payload
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
