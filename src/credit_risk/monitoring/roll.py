"""Stage 8.3: roll rates, cure rates and migration matrices.

A transition is a pair of consecutive months of the same loan (has_prev) whose
earlier month is active with a known bucket. The later month's state is its
exit type when the loan leaves (payoff / liquidation / other_exit), otherwise
its bucket, otherwise "unknown" (status XX). Counts and prev_upb weights per
(month, from, to) come from one DuckDB aggregation; everything else is
arithmetic on that small table:

    roll rates   current -> dpd30, dpd30 -> dpd60, dpd60 -> dpd90plus
    cure rates   dpd30 -> current, dpd60 -> current, dpd90plus -> current
    migration    row-normalised matrices for the reporting month and pooled
                 over each regime (pre_covid / covid / post_covid)

The reporting month's bucket-to-bucket counts are recomputed on a loan sample
with `transitions.transition_matrix`, the pandas reference that pairs months
by a merge instead of a window function.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.monitoring.buckets import BUCKET_ORDER
from credit_risk.monitoring.panel import PANEL_FILE
from credit_risk.monitoring.transitions import transition_matrix

TO_STATES = [*BUCKET_ORDER, "payoff", "liquidation", "other_exit", "unknown"]
ROLLS = (("current", "dpd30"), ("dpd30", "dpd60"), ("dpd60", "dpd90plus"))
CURES = (("dpd30", "current"), ("dpd60", "current"), ("dpd90plus", "current"))


def transition_counts(con: duckdb.DuckDBPyConnection, folder: Path) -> pd.DataFrame:
    panel = sql_str(folder / PANEL_FILE)
    df = con.execute(f"""
        SELECT period, prev_bucket AS from_state,
               coalesce(exit_type, bucket, 'unknown') AS to_state,
               count(*) AS loans, sum(prev_upb) AS upb
        FROM read_parquet({panel})
        WHERE has_prev AND prev_exit_type IS NULL AND prev_bucket IS NOT NULL
        GROUP BY ALL ORDER BY ALL
    """).df()
    df["period"] = pd.to_datetime(df["period"])
    df["upb"] = df["upb"].fillna(0.0)
    return df


def matrix(counts: pd.DataFrame, weight: str = "loans", normalize: bool = True) -> pd.DataFrame:
    m = counts.pivot_table(index="from_state", columns="to_state", values=weight, aggfunc="sum", fill_value=0)
    m = m.reindex(index=BUCKET_ORDER, columns=TO_STATES, fill_value=0)
    if not normalize:
        return m
    totals = m.sum(axis=1)
    return m.div(totals.where(totals > 0), axis=0)


def roll_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """One row per month: roll and cure rates by count and by UPB."""
    out = []
    for period, g in counts.groupby("period", sort=True):
        row = {"period": period}
        for w, suffix in (("loans", "count"), ("upb", "upb")):
            m = matrix(g, w)
            for a, b in (*ROLLS, *CURES):
                row[f"{a}_to_{b}_{suffix}"] = float(m.loc[a, b]) if a in m.index else np.nan
        out.append(row)
    return pd.DataFrame(out)


def roll_checks(con: duckdb.DuckDBPyConnection, folder: Path, counts: pd.DataFrame, month: pd.Timestamp,
                sample_loans: int, seed: int) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    panel = sql_str(folder / PANEL_FILE)
    # every transition out of an active known bucket lands in exactly one destination
    expected = con.execute(f"SELECT count(*) FROM read_parquet({panel}) "
                           f"WHERE has_prev AND prev_exit_type IS NULL AND prev_bucket IS NOT NULL").fetchone()[0]
    add("transitions counted = eligible month pairs", int(expected), int(counts["loans"].sum()), PANEL_FILE)
    add("destinations within the known states", 0, int((~counts["to_state"].isin(TO_STATES)).sum()), "transition_counts")
    rowsum = matrix(counts[counts["period"] == month]).sum(axis=1, min_count=1).dropna()   # empty rows stay NaN
    add(f"{month:%Y-%m} migration rows sum to 1", 0, int((rowsum.sub(1).abs() > 1e-12).sum()), "migration matrix")

    # pandas reference on a sample of loans: bucket-to-bucket counts for the pairs ending in the reporting month
    sample = con.execute(f"""
        SELECT loan_id, period, bucket FROM read_parquet({panel})
        WHERE loan_id IN (SELECT loan_id FROM (SELECT DISTINCT loan_id FROM read_parquet({panel})
                                               WHERE period = ? AND has_prev)
                                         USING SAMPLE reservoir({int(sample_loans)} ROWS) REPEATABLE ({int(seed)}))
          AND period BETWEEN ? AND ?
    """, [month.date(), (month - pd.DateOffset(months=1)).date(), month.date()]).df()
    reference = transition_matrix(sample, states=BUCKET_ORDER, normalize=False)
    ids = sample["loan_id"].unique().tolist()
    sql = con.execute(f"""
        SELECT prev_bucket AS from_state, bucket AS to_state, count(*) AS loans FROM read_parquet({panel})
        WHERE period = ? AND has_prev AND prev_bucket IS NOT NULL AND bucket IS NOT NULL AND loan_id IN (SELECT unnest(?::VARCHAR[]))
        GROUP BY ALL
    """, [month.date(), ids]).df()
    ours = sql.pivot_table(index="from_state", columns="to_state", values="loans", aggfunc="sum", fill_value=0) \
              .reindex(index=BUCKET_ORDER, columns=BUCKET_ORDER, fill_value=0)
    diff = int((ours.to_numpy() != reference.to_numpy()).sum())
    add(f"{month:%Y-%m} bucket transitions = transition_matrix on {len(ids):,} sampled loans", 0, diff, "transitions.py")
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    if x == 0 or x >= 0.0001:
        return f"{x:.2%}"
    return f"{x:.4%}" if x >= 0.000001 else "<0.0001%"


def matrix_markdown(m: pd.DataFrame, title: str) -> list[str]:
    lines = [f"### {title}", "", "| Từ \\ sang | " + " | ".join(m.columns) + " |", "|---|" + "---:|" * len(m.columns)]
    for state, r in m.iterrows():
        lines.append(f"| {state} | " + " | ".join(_pct(x) for x in r) + " |")
    return lines + [""]


def roll_markdown(counts: pd.DataFrame, rates: pd.DataFrame, regimes: pd.Series, month: pd.Timestamp, checks: pd.DataFrame) -> str:
    """`regimes` maps period -> regime (from the portfolio table)."""
    tagged = counts.assign(regime=counts["period"].map(regimes))
    trailing = rates[(rates["period"] < month)].tail(12)
    now = rates[rates["period"] == month].iloc[0]
    lines = ["## Roll rate và migration", "",
             "| Chuyển | Tháng báo cáo (số khoản) | TB 12 tháng trước | Tháng báo cáo (dư nợ) | TB 12 tháng trước |",
             "|---|---:|---:|---:|---:|"]
    for a, b in (*ROLLS, *CURES):
        c, u = f"{a}_to_{b}_count", f"{a}_to_{b}_upb"
        lines.append(f"| {a} → {b} | {_pct(now[c])} | {_pct(trailing[c].mean())} | {_pct(now[u])} | {_pct(trailing[u].mean())} |")
    lines.append("")
    lines += matrix_markdown(matrix(counts[counts["period"] == month]), f"Migration tháng {month:%Y-%m} (số khoản)")
    for regime, part in tagged.groupby("regime", sort=False):
        lines += matrix_markdown(matrix(part), f"Migration gộp giai đoạn {regime} (số khoản)")
    lines.append(f"Kiểm tra: {len(checks)}; lệch: {int((~checks['ok']).sum())}.")
    return "\n".join(lines) + "\n"
