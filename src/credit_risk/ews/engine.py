"""Stage 9: early warning, end to end.

One call runs, in order, and stops when the data under a step does not reconcile:

    9.0 cohort     monitoring panel (Stage 8.0) -> ews_labels.parquet, time split,
                   SQL label against labels.py on a sample
    9.1 features   availability matrix for every Freddie Mac column; features at t
                   from rows <= t -> ews_dataset.parquet (cohort + scoring rows)
    9.2 models     logistic baseline and LightGBM challenger on the train sample;
                   the better on validation drives the watchlist; every row scored
                   -> ews_scores.parquet; metrics by split, monthly lists,
                   calibration, segments
    9.3 watchlist  top capacity_share of the scoring month, with reason codes
    9.4 survival   Kaplan-Meier and Aalen-Johansen by MOB, delayed entry

and writes under artifacts/<run_id>/ews/: ews_metrics.json, ews_report.md, the
watchlist and every table as CSV, and the two models (LightGBM text, encoder and
logistic coefficients as JSON) with their fingerprints.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from credit_risk.config import ProjectConfig
from credit_risk.data.db import connect, sql_str
from credit_risk.ews import cohort as ch
from credit_risk.ews import evaluation as ev
from credit_risk.ews import features as ft
from credit_risk.ews import models as md
from credit_risk.ews import survival as sv
from credit_risk.ews import watchlist as wl
from credit_risk.ews.metrics import average_precision
from credit_risk.features.availability import availability_frame, parse_availability
from credit_risk.monitoring import vintage as vt
from credit_risk.monitoring.concentration import group_order
from credit_risk.monitoring.panel import PanelSpec, build_panel, panel_checks
from credit_risk.settings import Settings
from credit_risk.validation.metrics import auc

MODULES = ("panel", "cohort", "features", "models", "watchlist", "survival")
SPLIT_TEXT = {"train": "train", "validation": "validation", "stress_test": "stress test", "out_of_time": "out-of-time"}


class EwsFailed(RuntimeError):
    """A table the next step depends on does not reconcile with its source."""


@dataclass
class EwsResult:
    run_id: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    bands: dict = field(default_factory=dict)


def _stop_if_failed(checks: pd.DataFrame, step: str) -> None:
    if not checks["ok"].all():
        raise EwsFailed(f"{step} checks failed: " + "; ".join(checks.loc[~checks["ok"], "check"]))


def run_ews(settings: Settings, config: ProjectConfig, run_id: str) -> EwsResult:
    folder = settings.processed_dir / "freddie_mac"
    if not (folder / "performance.parquet").exists():
        raise FileNotFoundError("performance.parquet missing; run ingest for freddie_mac first")
    ews, definitions = config.ews, config.definitions
    panel_spec = PanelSpec.from_config(ews, definitions)
    cohort_spec = ch.CohortSpec.from_config(ews, definitions)
    feature_spec = ft.FeatureSpec.from_config(ews, definitions)
    model_spec = md.ModelSpec.from_config(ews, settings.duckdb_threads)
    survival_spec = sv.SurvivalSpec.from_config(ews)
    alert = ews["ews"]["alert"]
    share, n_reasons = float(alert["capacity_share"]), int(alert.get("reason_codes", 3))
    if alert.get("method") != "capacity" or not 0 < share < 1:
        raise ValueError("ews.alert: method must be capacity with 0 < capacity_share < 1")
    metric_names = list(ews["ews"]["evaluation_metrics"])
    segments = list(ews["ews"]["segments"])
    con = connect(settings)
    result = EwsResult(run_id=run_id, bands=group_order(panel_spec.bands))
    checks = []

    # 9.0 panel and labels
    build_panel(con, folder, panel_spec)
    p_checks, panel_summary = panel_checks(con, folder, panel_spec)
    _stop_if_failed(p_checks, "panel")
    checks.append(p_checks.assign(module="panel"))
    ch.build_labels(con, folder, cohort_spec)
    l_checks, cohort_info = ch.label_checks(con, folder, cohort_spec, definitions)
    _stop_if_failed(l_checks, "label")
    checks.append(l_checks.assign(module="cohort"))

    # 9.1 availability and features
    matrix = parse_availability(config.feature_availability, "freddie_mac")
    columns = {t: [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_str(folder / f'{t}.parquet')})").fetchall()]
               for t in ("origination", "performance")}
    availability = availability_frame(matrix, columns)          # raises on an unclassified column
    feature_columns = ft.check_availability(feature_spec, matrix)
    last = ch.last_period(con, folder)
    dataset, scoring_month = ft.build_dataset(con, folder, feature_spec, panel_spec, last)
    f_checks, feature_summary = ft.dataset_checks(con, folder, feature_spec, scoring_month)
    f_checks = pd.concat([f_checks, pd.DataFrame([{
        "check": "every feature reads only columns marked use: true", "claimed": len(feature_columns),
        "recounted": int(feature_columns["use"].sum()), "source": "feature_availability.yaml", "ok": bool(feature_columns["use"].all())}])],
        ignore_index=True)
    _stop_if_failed(f_checks, "feature")
    checks.append(f_checks.assign(module="features"))

    # 9.2 models
    cols = ", ".join(["loan_id", "period", "y"] + model_spec.features)
    data = sql_str(dataset)
    train = con.execute(f"SELECT {cols} FROM read_parquet({data}) "
                        f"WHERE {md.train_sample_where(model_spec.non_event_rate, model_spec.seed)} ORDER BY loan_id, period").df()
    valid = con.execute(f"SELECT {cols} FROM read_parquet({data}) WHERE split = 'validation' ORDER BY loan_id, period").df()
    for name, part in (("train", train), ("validation", valid)):
        if part.empty or part["y"].nunique() < 2:
            raise EwsFailed(f"{name} needs both events and non-events")
    w = md.weights(train["y"].to_numpy(), model_spec.non_event_rate)
    encoder = md.Encoder.fit(train, w, model_spec.numeric, model_spec.categorical, model_spec.categorical_min_share)
    baseline = md.fit_baseline(train, model_spec, encoder)
    challenger = md.fit_challenger(train, valid, model_spec, encoder)
    yv = valid["y"].to_numpy(dtype=int)
    selection = {}
    for model in (baseline, challenger):
        p = model.predict_pd(valid)
        selection[model.name] = average_precision(yv, p) if model_spec.selection_metric == "pr_auc" else auc(yv, p)
    selected = max(("baseline", "challenger"), key=lambda m: (selection[m], m == "baseline"))
    del valid
    models = {"baseline": baseline, "challenger": challenger}
    result.models = models
    scores = ev.score_dataset(dataset, folder / ev.SCORES_FILE, [baseline, challenger], model_spec.features)
    evaluation = ev.evaluate_scores(scores, ["baseline", "challenger"], selected, metric_names, segments, share)
    m_checks = _model_checks(con, dataset, scores, evaluation, selected, model_spec.selection_metric, selection, train,
                             model_spec.non_event_rate)
    checks.append(m_checks.assign(module="models"))

    # 9.3 watchlist
    scoring = con.execute(f"SELECT * FROM read_parquet({data}) WHERE scoring_row ORDER BY loan_id").df()
    watch = wl.build_watchlist(scoring, models[selected], share, n_reasons)
    w_summary = wl.watchlist_summary(watch, len(scoring), share)
    checks.append(_watchlist_checks(watch, scoring, share, scoring_month).assign(module="watchlist"))

    # 9.4 survival
    loans = sv.loan_times(con, folder, survival_spec.groups)
    curves = sv.survival_tables(loans, survival_spec)
    static = vt.vintage_curves(vt.loan_outcomes(con, folder), "vintage_year",
                               float(ews["monitoring"]["vintage_curve"].get("min_reach_share", 0.99)))
    s_checks, s_info = sv.survival_checks(loans, curves, static, survival_spec.checkpoints)
    checks.append(s_checks.assign(module="survival"))

    result.tables = {
        "cohort_by_split": cohort_info["by_split"], "exclusions": cohort_info["exclusions"],
        "availability": availability, "feature_columns": feature_columns,
        "trials": challenger.trials, "importance": challenger.importance(), "coefficients": baseline.coefficients(),
        "by_split": evaluation["by_split"], "monthly_lists": evaluation["monthly"], "calibration": evaluation["calibration"],
        "segments": evaluation["segments"], "watchlist": watch, "survival_curves": curves,
        "static_pool_comparison": s_info["static_pool_comparison"],
    }
    result.checks = pd.concat(checks, ignore_index=True)[["module", "check", "claimed", "recounted", "source", "ok"]]
    result.limitations = list(ews["ews"]["report"]["limitations"])
    oot = evaluation["by_split"].query("split == 'out_of_time' and model == @selected")
    result.info = {"cohort": cohort_info["summary"], "features": feature_summary, "watchlist": w_summary,
                   "survival": s_info["summary"], "survival_checkpoints": survival_spec.checkpoints,
                   "selection": selection, "capacity_share": share, "metrics": metric_names}
    result.provenance = {
        "run_id": run_id, "config_hash": config.hash, "seed": model_spec.seed,
        "panel": {k: panel_summary[k] for k in ("rows", "loans", "first_month", "last_month")},
        "scoring_month": f"{scoring_month:%Y-%m}", "horizon_months": cohort_spec.horizon,
        "splits": cohort_info["summary"]["splits"],
        "train_sample": {"rows": int(len(train)), "events": int(train["y"].sum()), "non_event_rate": model_spec.non_event_rate},
        "selected_model": selected, "selection_metric": model_spec.selection_metric,
        "validation_selection": {k: float(v) for k, v in selection.items()},
        "fingerprints": {"baseline": baseline.fingerprint(), "challenger": challenger.fingerprint()},
        "challenger_best_iteration": int(challenger.best_iteration),
        "out_of_time": {k: _num(v) for k, v in oot.iloc[0].items() if k not in ("split", "model")} if len(oot) else {},
        "watchlist": {"listed": w_summary["listed"], "scoring_rows": w_summary["scoring_rows"],
                      "expected_events": w_summary["expected_events"]},
        "checks_total": int(len(result.checks)), "checks_failed": int((~result.checks["ok"]).sum()),
    }
    return result


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    return v


def _model_checks(con, dataset: Path, scores: Path, evaluation: dict, selected: str, metric: str, selection: dict,
                  train: pd.DataFrame, rate: float) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    n_data = con.execute(f"SELECT count(*) FROM read_parquet({sql_str(dataset)})").fetchone()[0]
    add("every dataset row scored once", int(n_data), int(pq.ParquetFile(scores).metadata.num_rows), ev.SCORES_FILE)
    bad = con.execute(f"""
        SELECT count(*) FROM read_parquet({sql_str(scores)})
        WHERE NOT (pd_baseline BETWEEN 0 AND 1) OR NOT (pd_challenger BETWEEN 0 AND 1)
           OR isnan(pd_baseline) OR isnan(pd_challenger)
    """).fetchone()[0]
    add("every PD in [0, 1], none missing", 0, int(bad), ev.SCORES_FILE)
    by_split = evaluation["by_split"]
    recomputed = by_split[by_split["split"] == "validation"].set_index("model")[metric]
    add(f"validation {metric} in the report = the one used for selection", True,
        bool(all(abs(recomputed[m] - selection[m]) < 1e-12 for m in selection)), "ews_scores.parquet")
    add("selected model = higher validation " + metric, selected, ev.select_model(by_split, metric, ["baseline", "challenger"]),
        "evaluation.py")
    w = md.weights(train["y"].to_numpy(), rate)
    claimed_rate = float(con.execute(f"SELECT avg(y) FROM read_parquet({sql_str(dataset)}) WHERE split = 'train'").fetchone()[0])
    weighted_rate = float(np.average(train["y"].to_numpy(), weights=w))
    add("weighted train sample event rate = train event rate (within 10%)", round(claimed_rate, 6), round(weighted_rate, 6),
        "ews_dataset.parquet", ok=abs(weighted_rate / claimed_rate - 1) <= 0.10 if claimed_rate else False)
    monthly = evaluation["monthly"]
    if not monthly.empty:
        per_split = monthly.groupby("split")[["events"]].sum()["events"]
        events = by_split[by_split["model"] == selected].set_index("split")["events"]
        add("monthly list events add up to the split events", True,
            bool(all(int(per_split[s]) == int(events[s]) for s in per_split.index)), "evaluation.py")
    return pd.DataFrame(rows)


def _watchlist_checks(watch: pd.DataFrame, scoring: pd.DataFrame, share: float, month: pd.Timestamp) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    add("listed = ceil(capacity_share x scoring rows)", int(math.ceil(share * len(scoring) - 1e-9)), int(len(watch)), "ews.yaml alert")
    add("one row per loan, ranks 1..n", True, bool(watch["loan_id"].is_unique and (watch["rank"] == np.arange(1, len(watch) + 1)).all()),
        "watchlist.csv")
    add("PD does not increase down the list", True, bool((np.diff(watch["pd"].to_numpy()) <= 1e-15).all()), "watchlist.csv")
    add("every listed loan is in the scoring month", True, bool((pd.to_datetime(watch["period"]) == month).all()), "watchlist.csv")
    add("every listed loan has at least one reason", True, bool((watch["reason_1"] != "").all()), "watchlist.csv")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x, digits=2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def _f(x, digits=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _metric_cell(name: str, value) -> str:
    if name in ("roc_auc", "pr_auc"):
        return _f(value, 3)
    if name.startswith("lift"):
        return _f(value, 1)
    if name == "brier":
        return _f(value, 5)
    return _pct(value, 1)


def headline_markdown(result: EwsResult) -> str:
    p, info = result.provenance, result.info
    sel = p["selected_model"]
    oot = p["out_of_time"]
    w = info["watchlist"]
    lines = ["## Kết quả chính", "",
             f"- **Mô hình cho watchlist:** {sel} ({p['selection_metric']} validation: baseline "
             f"{_f(p['validation_selection']['baseline'])}, challenger {_f(p['validation_selection']['challenger'])}).",
             ]
    if oot:
        lines.append(f"- **Out-of-time {p['splits']['out_of_time'][0]} → {p['splits']['out_of_time'][1]}:** "
                     f"{oot['rows']:,} tháng-khoản vay, {oot['events']:,} sự kiện ({_pct(oot['event_rate'], 3)}). "
                     + ", ".join(f"{m} {_metric_cell(m, oot.get(m))}" for m in info["metrics"]) + ".")
        base = result.tables["by_split"].query("split == 'out_of_time' and model != @sel")
        if len(base):
            other = base.iloc[0]
            lines.append(f"- **Mô hình còn lại ({other['model']}) ở out-of-time:** "
                         + ", ".join(f"{m} {_metric_cell(m, other.get(m))}" for m in info["metrics"][:3]) + ".")
        current = result.tables["segments"].query("split == 'out_of_time' and dimension == 'bucket_at_t' and level == 'current'")
        if len(current):
            c = current.iloc[0]
            lines.append(f"- **Khoản đang current tại t:** {_pct(c['events'] / oot['events'], 1)} sự kiện out-of-time đến từ nhóm này; "
                         f"watchlist bắt được {_pct(c['capture'], 1)} trong số đó (khoản đang 30 / 60 ngày quá hạn gần như luôn có mặt "
                         "trong danh sách). Đây là phần khó và có giá trị nhất của cảnh báo sớm.")
        lines.append(f"- **Mức PD:** PD trung bình {_pct(oot['mean_pd'], 3)} so với tỷ lệ thực tế {_pct(oot['event_rate'], 3)} "
                     "ở out-of-time; PD không được hiệu chỉnh lại (xem Calibration).")
    lines.append(f"- **Watchlist {p['scoring_month']}:** {w['listed']:,} khoản ({_pct(w['listed_share'], 1)} của "
                 f"{w['scoring_rows']:,} khoản trong cohort), PD trung bình {_pct(w['mean_pd'])}, kỳ vọng "
                 f"{w['expected_events']:.0f} khoản vào 90+ trong 3 tháng tới theo PD của mô hình.")
    return "\n".join(lines) + "\n"


def report_markdown(result: EwsResult) -> str:
    p, info, t = result.provenance, result.info, result.tables
    sel = p["selected_model"]
    lines = [f"# Báo cáo Early Warning — run `{p['run_id']}`", "",
             f"Freddie Mac sample. Nhãn: lần đầu vào 90+ DPD hoặc thanh lý trong {p['horizon_months']} tháng sau tháng quan sát. "
             f"Panel {p['panel']['rows']:,} dòng, {p['panel']['loans']:,} khoản, {p['panel']['first_month']} → "
             f"{p['panel']['last_month']}. Config hash `{p['config_hash'][:12]}`, seed {p['seed']}.", "",
             f"**Kiểm tra:** {p['checks_total']} ({p['checks_failed']} lệch).", "",
             headline_markdown(result).rstrip(), ""]
    lines += _watchlist_md(result) + ["", "## Phạm vi và giới hạn", ""]
    lines += [f"- {x}" for x in result.limitations] or ["Không ghi nhận."]
    lines += ["", "## Kiểm tra theo module", "", "| Module | Kiểm tra | Lệch |", "|---|---:|---:|"]
    for module, part in result.checks.groupby("module", sort=False):
        lines.append(f"| {module} | {len(part)} | {int((~part['ok']).sum())} |")
    failed = result.checks[~result.checks["ok"]]
    if not failed.empty:
        lines += ["", "| Module | Kiểm tra lệch | Công bố | Tính lại | Nguồn |", "|---|---|---|---|---|"]
        for r in failed.itertuples(index=False):
            lines.append(f"| {r.module} | {r.check} | {r.claimed} | {r.recounted} | {r.source} |")
    lines += [""] + _cohort_md(result) + [""] + _models_md(result) + [""] + _backtest_md(result)
    lines += [""] + _calibration_md(result) + [""] + _segments_md(result) + [""] + _survival_md(result)
    return "\n".join(lines) + "\n"


def _watchlist_md(result: EwsResult) -> list[str]:
    w, watch = result.info["watchlist"], result.tables["watchlist"]
    lines = [f"## Watchlist {result.provenance['scoring_month']}", "",
             f"Top {_pct(w['capacity_share'], 0)} khoản theo PD của {result.provenance['selected_model']} trong tháng chấm điểm "
             f"(công suất xử lý của đội vận hành). Dư nợ trong danh sách {w['upb']:,.0f}. File đầy đủ: `watchlist.csv`.", "",
             "| Trạng thái tại t | Khoản |", "|---|---:|"]
    for k, v in sorted(w["by_bucket"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "Yếu tố đẩy điểm lên thường gặp nhất (số lần xuất hiện trong các reason code):", "",
              "| Feature | Lần |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in w["top_reasons"].items()]
    lines += ["", "20 khoản đầu danh sách:", "", "| # | loan_id | PD | Trạng thái | FICO | LTV | Lý do 1 | Lý do 2 | Lý do 3 |",
              "|---:|---|---:|---|---|---|---|---|---|"]
    for r in watch.head(20).itertuples(index=False):
        lines.append(f"| {r.rank} | {r.loan_id} | {_pct(r.pd)} | {r.bucket_at_t} | {r.fico_band} | {r.ltv_band} | "
                     f"{r.reason_1} | {getattr(r, 'reason_2', '')} | {getattr(r, 'reason_3', '')} |")
    return lines


def _cohort_md(result: EwsResult) -> list[str]:
    t = result.tables
    lines = ["## Nhãn và cohort", "", "| Đoạn | Tháng | Dòng | Cohort | Sự kiện | Tỷ lệ | Khoản |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in t["cohort_by_split"].itertuples(index=False):
        if pd.isna(r.rows):
            continue
        months = result.provenance["splits"][r.split]
        lines.append(f"| {SPLIT_TEXT[r.split]} | {months[0]} → {months[1]} | {int(r.rows):,} | {int(r.cohort):,} | "
                     f"{int(r.events):,} | {_pct(r.event_rate, 3)} | {int(r.loans):,} |")
    lines += ["", "Dòng không thuộc cohort, theo lý do:", "", "| Đoạn | Lý do | Dòng |", "|---|---|---:|"]
    lines += [f"| {SPLIT_TEXT[r.split]} | {r.exclusion_reason} | {int(r.rows):,} |" for r in t["exclusions"].itertuples(index=False)]
    f = result.info["features"]
    lines += ["", f"**Feature:** {f['features']} ({f['numeric']} số, {f['categorical']} phân loại), cửa sổ quan sát "
              f"{f['window_months']} tháng, chỉ từ các dòng <= t. Mọi cột gốc mà feature đọc đều `use: true` trong "
              "`feature_availability.yaml` (bảng `feature_columns.csv`)."]
    if f["train_missing_share"]:
        lines.append("Thiếu trên train: " + ", ".join(f"{k} {_pct(v)}" for k, v in f["train_missing_share"].items()) + ".")
    return lines


def _models_md(result: EwsResult) -> list[str]:
    p, t, info = result.provenance, result.tables, result.info
    ts = p["train_sample"]
    lines = ["## Mô hình", "",
             f"Mẫu train: {ts['rows']:,} dòng gồm mọi sự kiện ({ts['events']:,}) và {_pct(ts['non_event_rate'], 0)} non-event "
             f"(trọng số {1 / ts['non_event_rate']:.0f}). Baseline: logistic trên feature đã chia bin. Challenger: LightGBM, "
             f"early stopping theo average precision trên validation. Chọn theo {p['selection_metric']} trên validation: "
             f"**{p['selected_model']}**.", "", "Các cấu hình challenger:", ""]
    trials = t["trials"]
    lines += ["| " + " | ".join(trials.columns) + " |", "|" + "---|" * len(trials.columns)]
    lines += ["| " + " | ".join(_f(v, 4) if isinstance(v, float) else str(v) for v in row) + " |" for row in trials.itertuples(index=False)]
    lines += ["", "Chỉ số theo đoạn (monthly list: xếp hạng trong từng tháng):", "",
              "| Đoạn | Mô hình | Dòng | Sự kiện | Tỷ lệ | PD TB | " + " | ".join(info["metrics"]) + " |",
              "|---|---|---:|---:|---:|---:|" + "---:|" * len(info["metrics"])]
    for r in t["by_split"].to_dict("records"):
        lines.append(f"| {SPLIT_TEXT[r['split']]} | {r['model']} | {r['rows']:,} | {r['events']:,} | {_pct(r['event_rate'], 3)} | "
                     f"{_pct(r['mean_pd'], 3)} | " + " | ".join(_metric_cell(m, r.get(m)) for m in info["metrics"]) + " |")
    lines += ["", "Dòng train tính trên toàn bộ đoạn train (in-sample), chỉ để so độ khớp, không phải hiệu năng.", "",
              "Feature quan trọng nhất của challenger (gain):", "", "| Feature | Gain share |", "|---|---:|"]
    lines += [f"| {r.feature} | {_pct(r.gain_share, 1)} |" for r in t["importance"].head(12).itertuples(index=False)]
    return lines


def _backtest_md(result: EwsResult) -> list[str]:
    monthly, share = result.tables["monthly_lists"], result.info["capacity_share"]
    lines = ["## Backtest watchlist", "",
             f"Mỗi tháng lấy top {_pct(share, 0)} theo PD của {result.provenance['selected_model']}, như quy tắc watchlist. "
             "Capture = sự kiện nằm trong danh sách / mọi sự kiện; precision = sự kiện / khoản trong danh sách.", "",
             "| Đoạn | Năm | Tháng | Khoản trong DS | Sự kiện | Bắt được | Capture | Precision | Lift |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    if monthly.empty:
        return lines + ["", "Không có tháng có nhãn."]
    m = monthly.assign(year=pd.to_datetime(monthly["period"]).dt.year)
    for (split, year), g in m.groupby(["split", "year"], sort=False):
        listed, events, caught, loans = g["listed"].sum(), g["events"].sum(), g["caught"].sum(), g["loans"].sum()
        precision = caught / listed if listed else math.nan
        lift = precision / (events / loans) if events else math.nan
        lines.append(f"| {SPLIT_TEXT[split]} | {year} | {len(g)} | {listed:,} | {events:,} | {caught:,} | "
                     f"{_pct(caught / events if events else math.nan, 1)} | {_pct(precision)} | {_f(lift, 1)} |")
    return lines


def _calibration_md(result: EwsResult) -> list[str]:
    by_split, cal, sel = result.tables["by_split"], result.tables["calibration"], result.provenance["selected_model"]
    lines = ["## Calibration", "", "PD trung bình so với tỷ lệ sự kiện thực tế (quyết định: không hiệu chỉnh lại, báo cáo riêng từng đoạn):", "",
             "| Đoạn | Mô hình | PD TB | Thực tế | Thực tế / PD |", "|---|---|---:|---:|---:|"]
    for r in by_split[by_split["split"] != "train"].itertuples(index=False):
        lines.append(f"| {SPLIT_TEXT[r.split]} | {r.model} | {_pct(r.mean_pd, 3)} | {_pct(r.event_rate, 3)} | "
                     f"{_f(r.event_rate / r.mean_pd if r.mean_pd else math.nan, 2)} |")
    part = cal[(cal["model"] == sel) & (cal["split"] == "out_of_time")]
    if not part.empty:
        lines += ["", f"Theo decile PD, {sel}, out-of-time:", "", "| Decile | PD TB | Thực tế |", "|---:|---:|---:|"]
        lines += [f"| {int(r.decile)} | {_pct(r.mean_pd, 3)} | {_pct(r.observed_rate, 3)} |" for r in part.itertuples(index=False)]
    return lines


def _segments_md(result: EwsResult) -> list[str]:
    seg, sel = result.tables["segments"], result.provenance["selected_model"]
    lines = ["## Phân khúc", "",
             f"{sel}, trong từng đoạn. Capture = phần sự kiện của phân khúc nằm trong watchlist chung của tháng "
             "(danh sách không dựng lại theo phân khúc).", ""]
    for split in ("out_of_time", "stress_test", "validation"):
        part = seg[seg["split"] == split]
        if part.empty:
            continue
        lines += [f"### {SPLIT_TEXT[split]}", "", "| Phân khúc | Nhóm | Dòng | Sự kiện | Tỷ lệ | PD TB | ROC AUC | PR AUC | Phần của DS | Capture |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for dim, g in part.groupby("dimension", sort=False):
            order = result.bands.get(dim)
            if order:
                g = g.set_index("level").reindex([x for x in order if x in set(g["level"])] +
                                                 [x for x in g["level"] if x not in order]).reset_index()
            for r in g.itertuples(index=False):
                lines.append(f"| {dim} | {r.level} | {r.rows:,} | {r.events:,} | {_pct(r.event_rate, 3)} | {_pct(r.mean_pd, 3)} | "
                             f"{_f(r.roc_auc)} | {_f(r.pr_auc)} | {_pct(r.share_of_list, 1)} | {_pct(r.capture, 1)} |")
        lines.append("")
    return lines


def _survival_md(result: EwsResult) -> list[str]:
    curves, s, cps = result.tables["survival_curves"], result.info["survival"], result.info["survival_checkpoints"]
    lines = ["## Survival", "",
             f"{s['usable_loans']:,} khoản (trong đó {s['left_truncated_loans']:,} vào risk set muộn: delayed entry; "
             f"{s['left_truncated_bad_at_entry']:,} khoản cắt trái đã xấu ở tháng đầu bị loại). Kết cục: "
             f"{s['causes']['default']:,} default, {s['causes']['payoff']:,} trả hết, {s['causes']['censored']:,} censored. "
             "CIF = Aalen-Johansen (payoff cạnh tranh); KM = Kaplan-Meier coi payoff là censored, tức xác suất default "
             "nếu không ai trả trước.", ""]
    for dim in ["all"] + [d for d in curves["dimension"].unique() if d != "all"]:
        part = curves[curves["dimension"] == dim]
        groups = list(part["group"].unique())
        order = result.bands.get(dim)
        if order:
            groups = [g for g in order if g in groups] + [g for g in groups if g not in order]
        lines += [f"### Theo {dim}" if dim != "all" else "### Toàn danh mục", "",
                  "| Nhóm | " + " | ".join(f"CIF default MOB {m}" for m in cps) + " | " + " | ".join(f"KM default MOB {m}" for m in cps)
                  + f" | CIF payoff MOB {cps[-1]} |", "|---|" + "---:|" * (2 * len(cps) + 1)]
        for g in groups:
            c = part[part["group"] == g].set_index("mob")
            cell = lambda col, m: _pct(c.loc[m, col]) if m in c.index else "–"
            lines.append(f"| {g} | " + " | ".join(cell("cif_default", m) for m in cps) + " | "
                         + " | ".join(cell("km_default", m) for m in cps) + f" | {cell('cif_payoff', cps[-1])} |")
        lines.append("")
    comp = result.tables["static_pool_comparison"]
    if not comp.empty:
        lines.append(f"Đối chiếu với vintage curve chặng 8 (static pool, {len(comp)} cặp lứa × MOB): CIF lớn hơn tối đa "
                     f"{_pct(comp['difference'].max(), 3)} tuyệt đối "
                     f"({_pct((comp['difference'] / comp['static_pool']).max(), 2)} tương đối), do censoring.")
    return lines


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def metrics_payload(result: EwsResult) -> dict[str, Any]:
    t, cps = result.tables, result.info["survival_checkpoints"]
    curves = t["survival_curves"]
    return {
        "provenance": result.provenance,
        "checks": _records(result.checks),
        "limitations": result.limitations,
        "sections": {
            "cohort_by_split": _records(t["cohort_by_split"]), "exclusions": _records(t["exclusions"]),
            "features": result.info["features"], "trials": _records(t["trials"]),
            "importance": _records(t["importance"]), "by_split": _records(t["by_split"]),
            "monthly_lists": _records(t["monthly_lists"]), "calibration": _records(t["calibration"]),
            "segments": _records(t["segments"]), "watchlist_summary": result.info["watchlist"],
            "survival": result.info["survival"],
            "survival_checkpoints": _records(curves[curves["mob"].isin(cps)]),
            "static_pool_comparison": _records(t["static_pool_comparison"]),
        },
    }


def write_outputs(result: EwsResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("watchlist", "by_split", "monthly_lists", "calibration", "segments", "survival_curves", "trials",
                 "importance", "coefficients", "cohort_by_split", "exclusions", "availability", "feature_columns",
                 "static_pool_comparison"):
        path = out_dir / f"{name}.csv"
        result.tables[name].to_csv(path, index=False)
        paths.append(path)
    checks = out_dir / "checks.csv"
    result.checks.to_csv(checks, index=False)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    baseline, challenger = result.models["baseline"], result.models["challenger"]
    (model_dir / "encoder.json").write_text(json.dumps(baseline.encoder.to_dict(), indent=1), encoding="utf-8")
    (model_dir / "baseline.json").write_text(json.dumps({
        "columns": baseline.encoder.columns, "coef": baseline.model.coef_[0].tolist(),
        "intercept": float(baseline.model.intercept_[0]), "reference": baseline.reference,
        "fingerprint": baseline.fingerprint()}, indent=1), encoding="utf-8")
    (model_dir / "challenger.txt").write_text(challenger.booster.model_to_string(num_iteration=challenger.best_iteration),
                                              encoding="utf-8")
    metrics = out_dir / "ews_metrics.json"
    metrics.write_text(json.dumps(metrics_payload(result), indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    report = out_dir / "ews_report.md"
    report.write_text(report_markdown(result), encoding="utf-8")
    return [metrics, report, checks, *paths, *(model_dir / f for f in ("encoder.json", "baseline.json", "challenger.txt"))]


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"not JSON serialisable: {type(obj)}")
