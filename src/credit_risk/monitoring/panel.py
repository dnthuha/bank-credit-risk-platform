"""Stage 8.0: the monitoring panel, one row per loan and reporting month.

Every Stage 8 module reads this table instead of the 36-column raw panel. It is
built in DuckDB (35.6M rows do not fit in pandas on this laptop) and written to
`data/processed/freddie_mac/monitoring_panel.parquet`:

    loan_id, period, mob            reporting month and months on book (loan_age)
    vintage_year, vintage           from loan_id (never first_payment_date: ~16%
                                    of loans pay first in the next year)
    bucket                          current | dpd30 | dpd60 | dpd90plus (RA
                                    included) | NULL when the status is unknown
    exit_type                       payoff | liquidation | other_exit | NULL
    is_bad, first_bad               90+ or liquidation; the first such month
    upb, removal_upb                current_upb; UPB removed at zero balance
    has_prev                        TRUE when the loan has a row exactly one month
                                    earlier, FALSE otherwise (never NULL)
    prev_bucket, prev_upb,          that previous row's values, NULL when has_prev
    prev_exit_type                  is FALSE
    left_truncated                  first reported loan_age > max_first_loan_age
    fico_band, ltv_band, dti_band   bands from ews.yaml (missing -> "missing")
    orig_upb, property_state,       from origination (orig_upb weighs vintage curves)
    channel, harp

The SQL bucket mapping must agree with `buckets.status_to_bucket`, the pandas
reference the EWS label tests pin down. The build checks that on every distinct
status code in the data, so the two can never drift apart silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.monitoring.buckets import BUCKET_ORDER, status_to_bucket

PANEL_FILE = "monitoring_panel.parquet"
MISSING = "missing"


@dataclass(frozen=True)
class Band:
    label: str
    min: float | None = None
    max: float | None = None

    def contains(self, value) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


@dataclass(frozen=True)
class PanelSpec:
    bands: dict[str, list[Band]]                 # fico / ltv / dti -> ordered bands
    max_first_loan_age: int
    vintage_granularity: str                     # year | quarter
    codes: dict[str, Any]                        # definitions.mortgage.delinquency_status_codes
    zero_balance: dict[str, list[str]]           # definitions.mortgage.zero_balance_codes

    @classmethod
    def from_config(cls, ews: dict, definitions: dict) -> "PanelSpec":
        mon = ews["monitoring"]
        bands = {name: [Band(str(b["label"]), b.get("min"), b.get("max")) for b in spec]
                 for name, spec in mon["bands"].items()}
        for name, spec in bands.items():
            labels = [b.label for b in spec]
            if len(set(labels)) != len(labels) or MISSING in labels:
                raise ValueError(f"ews.monitoring.bands.{name}: labels must be unique and not '{MISSING}'")
        granularity = str(mon.get("vintage_granularity", "quarter"))
        if granularity not in ("year", "quarter"):
            raise ValueError(f"ews.monitoring.vintage_granularity must be year or quarter, got {granularity!r}")
        mortgage = definitions["mortgage"]
        return cls(bands=bands, max_first_loan_age=int(mon["vintage_curve"]["max_first_loan_age"]),
                   vintage_granularity=granularity, codes=mortgage["delinquency_status_codes"],
                   zero_balance=mortgage["zero_balance_codes"])


# ---------------------------------------------------------------------------
# SQL fragments generated from the config
# ---------------------------------------------------------------------------

def _in(column: str, values) -> str:
    return f"{column} IN ({', '.join(sql_str(str(v)) for v in values)})"


def bucket_sql(column: str, codes: dict) -> str:
    """The SQL twin of `status_to_bucket`."""
    return (f"CASE WHEN {_in(column, codes['current'])} THEN 'current' "
            f"WHEN {_in(column, codes['dpd30'])} THEN 'dpd30' "
            f"WHEN {_in(column, codes['dpd60'])} THEN 'dpd60' "
            f"WHEN TRY_CAST({column} AS INTEGER) >= {int(codes['dpd90plus_min_numeric_code'])} THEN 'dpd90plus' "
            f"WHEN {column} = {sql_str(codes['reo_acquisition'])} THEN 'dpd90plus' END")


def exit_sql(column: str, zero_balance: dict) -> str:
    return (f"CASE WHEN {_in(column, zero_balance['voluntary_payoff'])} THEN 'payoff' "
            f"WHEN {_in(column, zero_balance['liquidation'])} THEN 'liquidation' "
            f"WHEN {column} IS NOT NULL THEN 'other_exit' END")


def band_sql(column: str, bands: list[Band]) -> str:
    parts = []
    for b in bands:
        cond = [f"{column} >= {b.min}" if b.min is not None else None, f"{column} <= {b.max}" if b.max is not None else None]
        cond = " AND ".join(c for c in cond if c) or "TRUE"
        parts.append(f"WHEN {cond} THEN {sql_str(b.label)}")
    return f"CASE WHEN {column} IS NULL THEN '{MISSING}' {' '.join(parts)} ELSE '{MISSING}' END"


def assign_band(value, bands: list[Band]) -> str:
    """Python reference for `band_sql`."""
    if value is None or pd.isna(value):
        return MISSING
    return next((b.label for b in bands if b.contains(value)), MISSING)


def vintage_sql(granularity: str) -> str:
    year = "(2000 + CAST(substr(p.loan_id, 2, 2) AS INTEGER))"
    return year if granularity == "year" else f"CAST({year} AS VARCHAR) || substr(p.loan_id, 4, 2)"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_panel_sql(folder: Path, spec: PanelSpec) -> str:
    perf, orig = sql_str(folder / "performance.parquet"), sql_str(folder / "origination.parquet")
    return f"""
    WITH perf AS (
        SELECT p.loan_id, p.period, p.loan_age AS mob, p.current_upb AS upb,
               p.zero_balance_removal_upb AS removal_upb,
               nullif(trim(p.delinquency_status), '') AS status,
               nullif(trim(p.zero_balance_code), '') AS zb,
               (2000 + CAST(substr(p.loan_id, 2, 2) AS INTEGER)) AS vintage_year,
               {vintage_sql(spec.vintage_granularity)} AS vintage
        FROM read_parquet({perf}) p
    ), tagged AS (
        SELECT *, {bucket_sql('status', spec.codes)} AS bucket, {exit_sql('zb', spec.zero_balance)} AS exit_type
        FROM perf
    ), flagged AS (
        SELECT *, (coalesce(bucket = 'dpd90plus', FALSE) OR coalesce(exit_type = 'liquidation', FALSE)) AS is_bad
        FROM tagged
    ), windowed AS (
        SELECT *,
               min(CASE WHEN is_bad THEN period END) OVER w_loan AS first_bad_period,
               min(mob) OVER w_loan AS first_mob,
               lag(period) OVER w_time AS lag_period,
               lag(bucket) OVER w_time AS lag_bucket,
               lag(upb) OVER w_time AS lag_upb,
               lag(exit_type) OVER w_time AS lag_exit_type
        FROM flagged
        WINDOW w_loan AS (PARTITION BY loan_id), w_time AS (PARTITION BY loan_id ORDER BY period)
    )
    SELECT w.loan_id, w.period, w.mob, w.vintage_year, w.vintage, w.bucket, w.exit_type, w.is_bad,
           coalesce(w.period = w.first_bad_period, FALSE) AS first_bad,
           w.upb, w.removal_upb,
           coalesce(w.lag_period = w.period - INTERVAL 1 MONTH, FALSE) AS has_prev,
           CASE WHEN w.lag_period = w.period - INTERVAL 1 MONTH THEN w.lag_bucket END AS prev_bucket,
           CASE WHEN w.lag_period = w.period - INTERVAL 1 MONTH THEN w.lag_upb END AS prev_upb,
           CASE WHEN w.lag_period = w.period - INTERVAL 1 MONTH THEN w.lag_exit_type END AS prev_exit_type,
           (w.first_mob > {spec.max_first_loan_age}) AS left_truncated,
           {band_sql('o.fico', spec.bands['fico'])} AS fico_band,
           {band_sql('o.orig_ltv', spec.bands['ltv'])} AS ltv_band,
           {band_sql('o.orig_dti', spec.bands['dti'])} AS dti_band,
           o.orig_upb,
           coalesce(o.property_state, '{MISSING}') AS property_state,
           coalesce(o.channel, '{MISSING}') AS channel,
           coalesce(o.harp_indicator, '{MISSING}') AS harp
    FROM windowed w LEFT JOIN read_parquet({orig}) o USING (loan_id)
    """


def build_panel(con: duckdb.DuckDBPyConnection, folder: Path, spec: PanelSpec) -> Path:
    out = folder / PANEL_FILE
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({build_panel_sql(folder, spec)}) TO {sql_str(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Checks: the panel against its sources and against the pandas reference
# ---------------------------------------------------------------------------

def panel_checks(con: duckdb.DuckDBPyConnection, folder: Path, spec: PanelSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, perf, orig = (sql_str(folder / f) for f in (PANEL_FILE, "performance.parquet", "origination.parquet"))
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    n_perf, loans_perf = con.execute(f"SELECT count(*), count(DISTINCT loan_id) FROM read_parquet({perf})").fetchone()
    n_panel, loans_panel, dup = con.execute(
        f"SELECT count(*), count(DISTINCT loan_id), count(*) - count(DISTINCT (loan_id, period)) FROM read_parquet({panel})").fetchone()
    add("panel rows = performance rows", int(n_perf), int(n_panel), "performance.parquet")
    add("panel loans = performance loans", int(loans_perf), int(loans_panel), "performance.parquet")
    add("no duplicated (loan, month)", 0, int(dup), PANEL_FILE)
    no_orig = con.execute(f"SELECT count(DISTINCT loan_id) FROM read_parquet({panel}) WHERE loan_id NOT IN "
                          f"(SELECT loan_id FROM read_parquet({orig}))").fetchone()[0]
    add("every panel loan has an origination row", 0, int(no_orig), "origination.parquet")

    # vintage from loan_id against the file the loan came from (sample_orig_YYYY.txt)
    wrong_year = con.execute(f"""
        SELECT count(*) FROM read_parquet({orig})
        WHERE 2000 + CAST(substr(loan_id, 2, 2) AS INTEGER) <> CAST(regexp_extract(source_file, '(\\d{{4}})', 1) AS INTEGER)
    """).fetchone()[0]
    add("vintage year from loan_id = origination file year", 0, int(wrong_year), "origination.parquet source_file")

    # SQL buckets = pandas reference on every distinct status code present
    codes = con.execute(f"SELECT DISTINCT nullif(trim(delinquency_status), '') AS s FROM read_parquet({perf})").df()["s"]
    sql_buckets = con.execute(f"SELECT s, {bucket_sql('s', spec.codes)} AS b FROM (SELECT unnest(?::VARCHAR[]) AS s)",
                              [[c for c in codes if c is not None]]).df()
    ref = status_to_bucket(sql_buckets["s"], spec.codes).astype(object).fillna("<NA>").to_numpy()
    mismatched = int((sql_buckets["b"].astype(object).fillna("<NA>").to_numpy() != ref).sum())
    add(f"SQL bucket = status_to_bucket on {len(sql_buckets)} distinct codes", 0, mismatched, "buckets.py")

    # bands: SQL = Python reference on every distinct origination value
    for name, column in (("fico", "fico"), ("ltv", "orig_ltv"), ("dti", "orig_dti")):
        values = con.execute(f"SELECT DISTINCT {column} AS v, {band_sql(column, spec.bands[name])} AS b FROM read_parquet({orig})").df()
        wrong = sum(assign_band(v, spec.bands[name]) != b for v, b in zip(values["v"], values["b"]))
        add(f"SQL {name} band = Python reference", 0, int(wrong), "ews.yaml bands")

    first_bad_per_loan = con.execute(
        f"SELECT max(n) FROM (SELECT count(*) FILTER (WHERE first_bad) n FROM read_parquet({panel}) GROUP BY loan_id)").fetchone()[0]
    add("at most one first_bad month per loan", True, int(first_bad_per_loan or 0) <= 1, PANEL_FILE)

    summary = con.execute(f"""
        SELECT count(*) AS rows, count(DISTINCT loan_id) AS loans, min(period) AS first_month, max(period) AS last_month,
               count(DISTINCT loan_id) FILTER (WHERE left_truncated) AS left_truncated_loans,
               count(*) FILTER (WHERE bucket IS NULL AND exit_type IS NULL) AS unknown_status_rows,
               count(DISTINCT loan_id) FILTER (WHERE first_bad) AS loans_ever_bad
        FROM read_parquet({panel})
    """).df().iloc[0].to_dict()
    summary = {k: (v.strftime("%Y-%m") if hasattr(v, "strftime") else int(v)) for k, v in summary.items()}
    summary["left_truncated_share"] = summary["left_truncated_loans"] / summary["loans"] if summary["loans"] else 0.0
    summary["buckets"] = BUCKET_ORDER
    return pd.DataFrame(rows), summary
