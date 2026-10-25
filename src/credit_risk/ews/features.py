"""Stage 9.1: features at observation month t, from rows <= t only.

One DuckDB query over the raw performance file computes, for every loan-month,
the behaviour at t and over the observation window (definitions.windows.
observation_months, a RANGE frame ending at the current row), joins the
origination attributes and keeps:

  * the labelled cohort rows of the four segments (split set, y set);
  * the scoring rows: the scoring month, loans in the cohort at t (the label
    is not known yet, so their exclusion reason is the incomplete window).

The result is `data/processed/freddie_mac/ews_dataset.parquet`. Every feature
names the raw columns it reads; each must be `use: true` in the Freddie Mac
section of feature_availability.yaml, or the build stops. Window frames end at
CURRENT ROW, and a test changes every month after t to show the features at t
do not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.ews.cohort import INCOMPLETE_WINDOW, LABELS_FILE, SPLITS
from credit_risk.features.availability import AvailabilityMatrix
from credit_risk.monitoring.panel import PanelSpec, band_sql, bucket_sql

DATASET_FILE = "ews_dataset.parquet"
SCORING = "scoring"
NEVER_DELINQUENT = 999

# name -> (SQL over the windowed table `w` joined with origination `o`, raw columns read as (table, column))
_O, _P = "origination", "performance"
FEATURES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "fico": ("o.fico", ((_O, "fico"),)),
    "orig_ltv": ("o.orig_ltv", ((_O, "orig_ltv"),)),
    "orig_cltv": ("o.orig_cltv", ((_O, "orig_cltv"),)),
    "orig_dti": ("o.orig_dti", ((_O, "orig_dti"),)),
    "orig_interest_rate": ("o.orig_interest_rate", ((_O, "orig_interest_rate"),)),
    "orig_upb": ("o.orig_upb", ((_O, "orig_upb"),)),
    "orig_loan_term": ("o.orig_loan_term", ((_O, "orig_loan_term"),)),
    "mi_pct": ("o.mi_pct", ((_O, "mi_pct"),)),
    "num_borrowers": ("o.num_borrowers", ((_O, "num_borrowers"),)),
    "num_units": ("o.num_units", ((_O, "num_units"),)),
    "mob": ("w.mob", ((_P, "loan_age"),)),
    "bucket_num": ("w.bnum", ((_P, "delinquency_status"),)),
    "bucket_change_1m": ("CASE WHEN w.prev_period = w.period - INTERVAL 1 MONTH THEN w.bnum - w.prev_bnum ELSE 0 END",
                         ((_P, "delinquency_status"),)),
    "months_30plus_window": ("w.months_30plus_window", ((_P, "delinquency_status"),)),
    "max_bucket_window": ("w.max_bucket_window", ((_P, "delinquency_status"),)),
    "months_since_delinquent": (f"coalesce(date_diff('month', w.last_delinquent, w.period), {NEVER_DELINQUENT})",
                                ((_P, "delinquency_status"),)),
    "ever_90plus_before_t": ("CAST(l.ever_90plus_before_t AS INTEGER)", ((_P, "delinquency_status"),)),
    "history_months_window": ("w.history_months_window", ((_P, "loan_age"),)),
    "upb_ratio": ("w.upb / nullif(o.orig_upb, 0)", ((_P, "current_upb"), (_O, "orig_upb"))),
    "paydown_window": ("(w.upb_window_start - w.upb) / nullif(o.orig_upb, 0)", ((_P, "current_upb"), (_O, "orig_upb"))),
    "rate_change": ("w.cur_rate - o.orig_interest_rate", ((_P, "current_interest_rate"), (_O, "orig_interest_rate"))),
    "non_interest_bearing_share": ("coalesce(w.nib_upb / nullif(w.upb, 0), 0)",
                                   ((_P, "current_non_interest_bearing_upb"), (_P, "current_upb"))),
    "assistance_months_window": ("w.assistance_months_window", ((_P, "borrower_assistance_plan"),)),
    "modified": ("CAST(coalesce(w.mod_flag IN ('Y', 'P'), FALSE) AS INTEGER)", ((_P, "modification_flag"),)),
    "payment_deferral": ("CAST(coalesce(w.deferral_flag IN ('P', 'C'), FALSE) AS INTEGER)", ((_P, "payment_deferral_flag"),)),
    "disaster": ("CAST(coalesce(w.disaster_flag = 'Y', FALSE) AS INTEGER)", ((_P, "delinquency_due_to_disaster"),)),
    "occupancy_status": ("o.occupancy_status", ((_O, "occupancy_status"),)),
    "loan_purpose": ("o.loan_purpose", ((_O, "loan_purpose"),)),
    "channel": ("o.channel", ((_O, "channel"),)),
    "property_type": ("o.property_type", ((_O, "property_type"),)),
    "first_time_homebuyer": ("o.first_time_homebuyer", ((_O, "first_time_homebuyer"),)),
    "harp_indicator": ("o.harp_indicator", ((_O, "harp_indicator"),)),
    "property_state": ("o.property_state", ((_O, "property_state"),)),
    "assistance_plan_at_t": ("coalesce(w.bap, 'none')", ((_P, "borrower_assistance_plan"),)),
}
# columns carried next to the features: keys, label, segments for the evaluation
CONTEXT = ["loan_id", "period", "split", "regime", "y", "event_month_offset", "bucket_at_t",
           "vintage_year", "fico_band", "ltv_band", "upb"]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]
    window_months: int
    categorical_min_share: float
    scoring_month: pd.Timestamp | None          # None = last month of the panel

    @classmethod
    def from_config(cls, ews: dict, definitions: dict) -> "FeatureSpec":
        cfg = ews["ews"]["features"]
        numeric, categorical = list(cfg["numeric"]), list(cfg["categorical"])
        unknown = [f for f in numeric + categorical if f not in FEATURES]
        if unknown:
            raise ValueError(f"ews.features: unknown features {unknown}; known: {sorted(FEATURES)}")
        if len(set(numeric + categorical)) != len(numeric + categorical):
            raise ValueError("ews.features: a feature is listed twice")
        month = ews["ews"].get("scoring_month")
        return cls(numeric=numeric, categorical=categorical, window_months=int(definitions["windows"]["observation_months"]),
                   categorical_min_share=float(cfg.get("categorical_min_share", 0.001)),
                   scoring_month=pd.Timestamp(f"{month}-01") if month else None)

    @property
    def names(self) -> list[str]:
        return self.numeric + self.categorical

    def raw_columns(self) -> list[tuple[str, str]]:
        return sorted({c for f in self.names for c in FEATURES[f][1]})


def check_availability(spec: FeatureSpec, matrix: AvailabilityMatrix) -> pd.DataFrame:
    """One row per (feature, raw column); raises when a column read is not usable."""
    rows = []
    for f in spec.names:
        for table, column in FEATURES[f][1]:
            c = matrix.table(table).resolve(column)
            rows.append({"feature": f, "table": table, "column": column, "availability": c.availability, "use": c.use})
    frame = pd.DataFrame(rows)
    blocked = frame[~frame["use"]]
    if not blocked.empty:
        raise ValueError("features read columns that feature_availability.yaml does not allow: "
                         + ", ".join(f"{r.feature} <- {r.table}.{r.column} ({r.availability})" for r in blocked.itertuples()))
    return frame


def dataset_sql(folder: Path, spec: FeatureSpec, panel_spec: PanelSpec, scoring_month: pd.Timestamp) -> str:
    perf, orig, labels = (sql_str(folder / f) for f in ("performance.parquet", "origination.parquet", LABELS_FILE))
    w = spec.window_months - 1
    features = ",\n           ".join(f"{FEATURES[f][0]} AS {f}" for f in spec.names)
    return f"""
    WITH raw AS (
        SELECT loan_id, period, loan_age AS mob, current_upb AS upb, current_interest_rate AS cur_rate,
               current_non_interest_bearing_upb AS nib_upb,
               nullif(trim(modification_flag), '') AS mod_flag, nullif(trim(payment_deferral_flag), '') AS deferral_flag,
               nullif(trim(delinquency_due_to_disaster), '') AS disaster_flag,
               nullif(trim(borrower_assistance_plan), '') AS bap,
               {bucket_sql("nullif(trim(delinquency_status), '')", panel_spec.codes)} AS bucket
        FROM read_parquet({perf})
    ), num AS (
        SELECT *, CASE bucket WHEN 'current' THEN 0 WHEN 'dpd30' THEN 1 WHEN 'dpd60' THEN 2 WHEN 'dpd90plus' THEN 3 END AS bnum
        FROM raw
    ), w AS (
        SELECT *,
               lag(bnum) OVER t AS prev_bnum, lag(period) OVER t AS prev_period,
               sum(CASE WHEN bnum >= 1 THEN 1 ELSE 0 END) OVER win AS months_30plus_window,
               max(bnum) OVER win AS max_bucket_window,
               count(*) OVER win AS history_months_window,
               first_value(upb) OVER win AS upb_window_start,
               sum(CASE WHEN bap IS NOT NULL THEN 1 ELSE 0 END) OVER win AS assistance_months_window,
               max(CASE WHEN bnum >= 1 THEN period END) OVER (t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_delinquent
        FROM num
        WINDOW t AS (PARTITION BY loan_id ORDER BY period),
               win AS (PARTITION BY loan_id ORDER BY period RANGE BETWEEN INTERVAL {w} MONTH PRECEDING AND CURRENT ROW)
    ), l AS (
        SELECT *, period = DATE '{scoring_month:%Y-%m-%d}' AS scoring_row FROM read_parquet({labels})
        WHERE (split IS NOT NULL AND y IS NOT NULL)
           OR (period = DATE '{scoring_month:%Y-%m-%d}'
               AND (exclusion_reason IS NULL OR exclusion_reason = '{INCOMPLETE_WINDOW}'))
    )
    SELECT l.loan_id, l.period,
           CASE WHEN l.scoring_row AND l.y IS NULL THEN '{SCORING}' ELSE l.split END AS split,
           l.regime, l.y, l.event_month_offset, l.bucket_at_t, l.scoring_row,
           (2000 + CAST(substr(l.loan_id, 2, 2) AS INTEGER)) AS vintage_year,
           {band_sql('o.fico', panel_spec.bands['fico'])} AS fico_band,
           {band_sql('o.orig_ltv', panel_spec.bands['ltv'])} AS ltv_band,
           w.upb,
           {features}
    FROM l JOIN w USING (loan_id, period) LEFT JOIN read_parquet({orig}) o USING (loan_id)
    """


def build_dataset(con: duckdb.DuckDBPyConnection, folder: Path, spec: FeatureSpec, panel_spec: PanelSpec,
                  last_period: pd.Timestamp) -> tuple[Path, pd.Timestamp]:
    scoring_month = spec.scoring_month or last_period
    if scoring_month > last_period:
        raise ValueError(f"ews.scoring_month {scoring_month:%Y-%m} is after the last month in the data ({last_period:%Y-%m})")
    out = folder / DATASET_FILE
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({dataset_sql(folder, spec, panel_spec, scoring_month)}) TO {sql_str(tmp)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out)
    return out, scoring_month


def dataset_checks(con: duckdb.DuckDBPyConnection, folder: Path, spec: FeatureSpec,
                   scoring_month: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    data, labels = sql_str(folder / DATASET_FILE), sql_str(folder / LABELS_FILE)
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    claimed = con.execute(f"""
        SELECT split, count(*), sum(y) FROM read_parquet({labels}) WHERE split IS NOT NULL AND y IS NOT NULL GROUP BY split
    """).df().set_index("split")
    got = con.execute(f"""
        SELECT split, count(*), sum(y) FROM read_parquet({data}) WHERE split <> '{SCORING}' GROUP BY split
    """).df().set_index("split")
    for split in [s for s in SPLITS if s in claimed.index]:  # fixed order: GROUP BY output order is not stable
        add(f"{split}: dataset rows = labelled cohort rows", int(claimed.loc[split].iloc[0]),
            int(got.loc[split].iloc[0]) if split in got.index else 0, LABELS_FILE)
        add(f"{split}: dataset events = labelled events", int(claimed.loc[split].iloc[1]),
            int(got.loc[split].iloc[1]) if split in got.index else 0, LABELS_FILE)
    dup = con.execute(f"SELECT count(*) - count(DISTINCT (loan_id, period)) FROM read_parquet({data})").fetchone()[0]
    add("no duplicated (loan, month)", 0, int(dup), DATASET_FILE)
    n_scoring, bad_scoring = con.execute(f"""
        SELECT count(*) FILTER (WHERE scoring_row),
               count(*) FILTER (WHERE scoring_row AND (bucket_at_t IS NULL OR bucket_at_t = 'dpd90plus'))
        FROM read_parquet({data})
    """).fetchone()
    add("scoring rows exist for the scoring month", True, int(n_scoring) > 0, DATASET_FILE)
    add("no scoring row is 90+ or unknown at t", 0, int(bad_scoring), DATASET_FILE)
    cohort_bad = con.execute(f"""
        SELECT count(*) FROM read_parquet({data}) WHERE bucket_num IS NULL OR bucket_num > 2
    """).fetchone()[0] if "bucket_num" in spec.names else 0
    add("bucket_num at t is 0, 1 or 2 on every row", 0, int(cohort_bad), DATASET_FILE)
    missing = con.execute(f"""
        SELECT {', '.join(f"avg(CASE WHEN {f} IS NULL THEN 1.0 ELSE 0.0 END) AS {f}" for f in spec.names)}
        FROM read_parquet({data}) WHERE split = 'train'
    """).df().iloc[0]
    summary = {"scoring_month": f"{scoring_month:%Y-%m}", "scoring_rows": int(n_scoring),
               "features": len(spec.names), "numeric": len(spec.numeric), "categorical": len(spec.categorical),
               "window_months": spec.window_months,
               "train_missing_share": {k: float(v) for k, v in missing.items() if v > 0}}
    return pd.DataFrame(rows), summary
