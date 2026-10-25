"""Stage 9.0: the three-month label on the whole panel, and the time split.

`labels.build_ews_labels` is the pandas reference the unit tests pin down; it
cannot hold 35.6M rows on this laptop. `label_sql` is its DuckDB twin over the
monitoring panel, written to `data/processed/freddie_mac/ews_labels.parquet`:

    loan_id, period                 observation month t
    split                           train | validation | stress_test | out_of_time
                                    from ews.time_split; NULL between segments and
                                    in the last `horizon` months (no full window)
    bucket_at_t, ever_90plus_before_t
    y, event_month_offset           1 / 0 and the month k of the event, as in labels.py
    exclusion_reason                why the row has no label (NULL when y is set)

The build re-labels a seeded sample of loans with the pandas reference from the
raw performance file and requires an exact match, so the two cannot drift apart.

Each segment must end `horizon` months before the next one starts, so that no
label window reaches into the following segment; the config is refused otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.ews.labels import (
    ALREADY_90PLUS,
    INACTIVE_AT_T,
    INCOMPLETE_WINDOW,
    PRIOR_90PLUS,
    STATUS_NOT_AVAILABLE,
    build_ews_labels,
)
from credit_risk.monitoring.panel import PANEL_FILE

LABELS_FILE = "ews_labels.parquet"
SPLITS = ("train", "validation", "stress_test", "out_of_time")


@dataclass(frozen=True)
class CohortSpec:
    horizon: int
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]]   # None = last month with a full window
    prior_90plus_excluded: bool
    reference_sample_loans: int
    seed: int
    covid_start: pd.Timestamp
    covid_end: pd.Timestamp

    @classmethod
    def from_config(cls, ews: dict, definitions: dict) -> "CohortSpec":
        cfg = ews["ews"]
        horizon = int(definitions["windows"]["performance_months"])
        raw = cfg["time_split"]
        if list(raw) != list(SPLITS):
            raise ValueError(f"ews.time_split must list {list(SPLITS)} in that order, got {list(raw)}")
        splits = {}
        for name in SPLITS:
            start = pd.Timestamp(f"{raw[name]['start']}-01")
            end = pd.Timestamp(f"{raw[name]['end']}-01") if raw[name].get("end") else None
            if end is not None and end < start:
                raise ValueError(f"ews.time_split.{name}: end before start")
            splits[name] = (start, end)
        for (a, (_, end)), (b, (start, _)) in zip(list(splits.items()), list(splits.items())[1:]):
            if end is None or end + pd.DateOffset(months=horizon) >= start:
                raise ValueError(f"ews.time_split: {a} must end at least {horizon} months before {b} starts, "
                                 "so no label window reaches into it")
        covid = definitions["covid_period"]
        return cls(horizon=horizon, splits=splits,
                   prior_90plus_excluded=not definitions["mortgage"].get("redefault_after_cure_counts_as_event", True),
                   reference_sample_loans=int(cfg.get("reference_sample_loans", 20000)), seed=int(ews.get("seed", 0)),
                   covid_start=pd.Timestamp(f"{covid['start']}-01"), covid_end=pd.Timestamp(f"{covid['end']}-01"))

    def split_bounds(self, last_period: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        last_labelled = last_period - pd.DateOffset(months=self.horizon)
        return {k: (s, e if e is not None else last_labelled) for k, (s, e) in self.splits.items()}


def _date(ts: pd.Timestamp) -> str:
    return f"DATE '{ts:%Y-%m-%d}'"


def split_sql(column: str, bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    parts = [f"WHEN {column} BETWEEN {_date(s)} AND {_date(e)} THEN '{k}'" for k, (s, e) in bounds.items()]
    return f"CASE {' '.join(parts)} END"


def regime_sql(column: str, spec: CohortSpec) -> str:
    return (f"CASE WHEN {column} < {_date(spec.covid_start)} THEN 'pre_covid' "
            f"WHEN {column} <= {_date(spec.covid_end)} THEN 'covid' ELSE 'post_covid' END")


def label_sql(panel: str, spec: CohortSpec, bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    """The SQL twin of `build_ews_labels` over the monitoring panel."""
    h = spec.horizon
    leads = ",\n".join(f"lead(period, {j}) OVER w AS p{j}, lead(state, {j}) OVER w AS s{j}" for j in range(1, h + 1))
    # state of month t+k: a row exactly k months later, among the next k rows (periods only increase)
    at_k = ",\n".join(
        "CASE " + " ".join(f"WHEN p{j} = period + INTERVAL {k} MONTH THEN s{j}" for j in range(1, k + 1)) + f" END AS m{k}"
        for k in range(1, h + 1))
    # the first informative month decides: B event, P payoff, missing / X (other exit, unknown) incomplete, N go on
    y_case, off_case, open_ = [], [], []
    for k in range(1, h + 1):
        still = " AND ".join(open_) or "TRUE"
        y_case.append(f"WHEN {still} AND m{k} = 'B' THEN 1 WHEN {still} AND m{k} = 'P' THEN 0 "
                      f"WHEN {still} AND (m{k} IS NULL OR m{k} = 'X') THEN NULL")
        off_case.append(f"WHEN {still} AND m{k} = 'B' THEN {k}")
        open_.append(f"m{k} = 'N'")
    exclusions = [("terminated", INACTIVE_AT_T), ("bucket = 'dpd90plus'", ALREADY_90PLUS), ("bucket IS NULL", STATUS_NOT_AVAILABLE)]
    if spec.prior_90plus_excluded:
        exclusions.append(("ever_bad_before", PRIOR_90PLUS))
    excl = " ".join(f"WHEN {cond} THEN '{label}'" for cond, label in exclusions)
    return f"""
    WITH base AS (
        SELECT loan_id, period, bucket, exit_type,
               CASE WHEN is_bad THEN 'B' WHEN exit_type = 'payoff' THEN 'P'
                    WHEN exit_type IS NOT NULL OR bucket IS NULL THEN 'X' ELSE 'N' END AS state,
               coalesce(period > min(CASE WHEN is_bad THEN period END) OVER l, FALSE) AS ever_bad_before,
               coalesce(period >= min(CASE WHEN exit_type IS NOT NULL THEN period END) OVER l, FALSE) AS terminated
        FROM read_parquet({panel})
        WINDOW l AS (PARTITION BY loan_id)
    ), led AS (
        SELECT *, {leads}
        FROM base WINDOW w AS (PARTITION BY loan_id ORDER BY period)
    ), win AS (
        SELECT *, {at_k}, CASE {excl} END AS cohort_exclusion FROM led
    ), labelled AS (
        SELECT *,
               CASE WHEN cohort_exclusion IS NOT NULL THEN NULL {' '.join(y_case)} ELSE 0 END AS y,
               CASE WHEN cohort_exclusion IS NOT NULL THEN NULL {' '.join(off_case)} END AS event_month_offset
        FROM win
    )
    SELECT loan_id, period, {split_sql('period', bounds)} AS split, {regime_sql('period', spec)} AS regime,
           bucket AS bucket_at_t, ever_bad_before AS ever_90plus_before_t,
           CAST(y AS TINYINT) AS y, CAST(event_month_offset AS TINYINT) AS event_month_offset,
           coalesce(cohort_exclusion, CASE WHEN y IS NULL THEN '{INCOMPLETE_WINDOW}' END) AS exclusion_reason
    FROM labelled
    """


def last_period(con: duckdb.DuckDBPyConnection, folder: Path) -> pd.Timestamp:
    return pd.Timestamp(con.execute(f"SELECT max(period) FROM read_parquet({sql_str(folder / PANEL_FILE)})").fetchone()[0])


def build_labels(con: duckdb.DuckDBPyConnection, folder: Path, spec: CohortSpec) -> Path:
    bounds = spec.split_bounds(last_period(con, folder))
    out = folder / LABELS_FILE
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({label_sql(sql_str(folder / PANEL_FILE), spec, bounds)}) TO {sql_str(tmp)} "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def reference_labels(con: duckdb.DuckDBPyConnection, folder: Path, spec: CohortSpec, definitions: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pandas labels, SQL labels) for the same seeded sample of loans, sorted alike."""
    perf, labels = sql_str(folder / "performance.parquet"), sql_str(folder / LABELS_FILE)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ews_ref_loans AS
        SELECT loan_id FROM (SELECT DISTINCT loan_id FROM read_parquet({perf}) ORDER BY loan_id)
        USING SAMPLE reservoir({spec.reference_sample_loans} ROWS) REPEATABLE ({spec.seed})
    """)
    raw = con.execute(f"""
        SELECT loan_id, period, delinquency_status, zero_balance_code FROM read_parquet({perf})
        WHERE loan_id IN (SELECT loan_id FROM ews_ref_loans)
    """).df()
    sql = con.execute(f"""
        SELECT loan_id, period, bucket_at_t, ever_90plus_before_t, y, event_month_offset, exclusion_reason
        FROM read_parquet({labels}) WHERE loan_id IN (SELECT loan_id FROM ews_ref_loans)
    """).df()
    ref = build_ews_labels(raw, definitions, horizon=spec.horizon)
    key = ["loan_id", "period"]
    ref["period"], sql["period"] = pd.to_datetime(ref["period"]), pd.to_datetime(sql["period"])
    return ref.sort_values(key).reset_index(drop=True), sql.sort_values(key).reset_index(drop=True)


def _canonical(value) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return "<NA>"
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _same(a: pd.Series, b: pd.Series) -> int:
    """Rows where two columns differ; NA equals NA and 1 equals 1.0 (Int8 against float with NaN)."""
    return int(sum(_canonical(x) != _canonical(y) for x, y in zip(a.tolist(), b.tolist())))


def label_checks(con: duckdb.DuckDBPyConnection, folder: Path, spec: CohortSpec,
                 definitions: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, labels = sql_str(folder / PANEL_FILE), sql_str(folder / LABELS_FILE)
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    n_panel = con.execute(f"SELECT count(*) FROM read_parquet({panel})").fetchone()[0]
    n_labels, both, neither, bad_offset = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE y IS NOT NULL AND exclusion_reason IS NOT NULL),
               count(*) FILTER (WHERE y IS NULL AND exclusion_reason IS NULL),
               count(*) FILTER (WHERE (y = 1) <> coalesce(event_month_offset BETWEEN 1 AND {spec.horizon}, FALSE))
        FROM read_parquet({labels})
    """).fetchone()
    add("label rows = panel rows", int(n_panel), int(n_labels), PANEL_FILE)
    add("a row has a label or an exclusion reason, never both", 0, int(both) + int(neither), LABELS_FILE)
    add("y = 1 exactly when the event month is in 1..horizon", 0, int(bad_offset), LABELS_FILE)

    # every event points at a bad panel month; every 0 has no bad month in its window
    wrong_events = con.execute(f"""
        SELECT count(*) FROM read_parquet({labels}) l
        LEFT JOIN (SELECT loan_id, period FROM read_parquet({panel}) WHERE is_bad) b
          ON b.loan_id = l.loan_id AND b.period = l.period + to_months(CAST(l.event_month_offset AS INTEGER))
        WHERE l.y = 1 AND b.loan_id IS NULL
    """).fetchone()[0]
    # the next `horizon` rows hold every month in (t, t + horizon]: periods only increase
    missed = con.execute(f"""
        WITH nb AS (
            SELECT loan_id, period, min(CASE WHEN is_bad THEN period END) OVER (
                       PARTITION BY loan_id ORDER BY period ROWS BETWEEN 1 FOLLOWING AND {spec.horizon} FOLLOWING) AS next_bad
            FROM read_parquet({panel}))
        SELECT count(*) FROM read_parquet({labels}) l JOIN nb USING (loan_id, period)
        WHERE l.y = 0 AND nb.next_bad <= l.period + INTERVAL {spec.horizon} MONTH
    """).fetchone()[0]
    add("every event points at a 90+ / liquidation month", 0, int(wrong_events), PANEL_FILE)
    add("no y = 0 row has a bad month in its window", 0, int(missed), PANEL_FILE)

    last = last_period(con, folder)
    bounds = spec.split_bounds(last)
    late = con.execute(f"""
        SELECT count(*) FROM read_parquet({labels})
        WHERE split IS NOT NULL AND period + INTERVAL {spec.horizon} MONTH > {_date(last)}
    """).fetchone()[0]
    add("no split row lacks a full window in the data", 0, int(late), LABELS_FILE)

    ref, sql = reference_labels(con, folder, spec, definitions)
    add(f"SQL rows = pandas rows on {spec.reference_sample_loans:,} sampled loans", len(ref), len(sql), "labels.py")
    if len(ref) == len(sql):
        for col in ("bucket_at_t", "ever_90plus_before_t", "y", "event_month_offset", "exclusion_reason"):
            add(f"SQL {col} = labels.py on the sample", 0, _same(ref[col], sql[col]), "labels.py")

    by_split = con.execute(f"""
        SELECT split, count(*) AS rows, count(y) AS cohort, sum(y) AS events, avg(y) AS event_rate,
               count(DISTINCT loan_id) AS loans, min(period) AS first_month, max(period) AS last_month
        FROM read_parquet({labels}) WHERE split IS NOT NULL GROUP BY split
    """).df().set_index("split").reindex(list(SPLITS)).reset_index()
    exclusions = con.execute(f"""
        SELECT split, exclusion_reason, count(*) AS rows FROM read_parquet({labels})
        WHERE split IS NOT NULL AND exclusion_reason IS NOT NULL GROUP BY ALL ORDER BY split, rows DESC
    """).df()
    summary = {"last_period": f"{last:%Y-%m}", "horizon": spec.horizon,
               "splits": {k: [f"{s:%Y-%m}", f"{e:%Y-%m}"] for k, (s, e) in bounds.items()},
               "reference_sample_loans": spec.reference_sample_loans}
    return pd.DataFrame(rows), {"summary": summary, "by_split": by_split, "exclusions": exclusions}
