"""Stage 8.2: vintage curves, cumulative default and payoff by months on book.

Static pool per vintage (decision 2026-10-25): loans observed from the start
(left_truncated = FALSE). For each vintage and MOB k,

    cum_default_rate_count(k) = loans whose first default (90+ or liquidation)
                                happened at MOB <= k  /  loans in the pool
    cum_default_rate_upb(k)   = the same with orig_upb as the weight

and likewise for voluntary payoff. Loans that pay off stay in the
denominator: this is the gross static-pool curve, so default and payoff
compete and their curves can be read side by side.

Right censoring: a curve stops at the last MOB that at least
`min_reach_share` of its pool could have reached by the panel's last month,
so the tail is pulled down by at most 1 - min_reach_share (relative) by
loans that are simply too young. Not every loan (share 1.0): a handful per
vintage start at loan_age 0 a year or more after the quarter in their
loan_id and would cut whole curves short. MOB is loan_age, never a row count.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.monitoring.panel import PANEL_FILE

CHECKPOINTS = (6, 12, 24, 36, 60)


def loan_outcomes(con: duckdb.DuckDBPyConnection, folder: Path) -> pd.DataFrame:
    """One row per loan: vintage, pool membership, MOB of first default / payoff, last reachable MOB."""
    panel = sql_str(folder / PANEL_FILE)
    return con.execute(f"""
        WITH last AS (SELECT max(period) AS last_period FROM read_parquet({panel}))
        SELECT loan_id, any_value(vintage) AS vintage, any_value(vintage_year) AS vintage_year,
               any_value(orig_upb) AS orig_upb, bool_or(left_truncated) AS left_truncated,
               min(mob) FILTER (WHERE first_bad) AS default_mob,
               min(mob) FILTER (WHERE exit_type = 'payoff') AS payoff_mob,
               arg_min(mob, period) + date_diff('month', min(period), any_value(last.last_period)) AS reachable_mob
        FROM read_parquet({panel}), last
        GROUP BY loan_id
    """).df()


def max_reported_mob(reachable: pd.Series, min_reach_share: float) -> int:
    """Largest MOB that at least `min_reach_share` of the pool could have reached."""
    return int(np.floor(reachable.quantile(1 - min_reach_share, interpolation="lower")))


def vintage_curves(loans: pd.DataFrame, by: str = "vintage", min_reach_share: float = 0.99) -> pd.DataFrame:
    """Long table: one row per (vintage, MOB) up to the vintage's reported MOB."""
    if not 0 < min_reach_share <= 1:
        raise ValueError(f"min_reach_share must be in (0, 1], got {min_reach_share}")
    pool = loans[~loans["left_truncated"]]
    out = []
    for vintage, g in pool.groupby(by, sort=True):
        max_mob = max_reported_mob(g["reachable_mob"], min_reach_share)
        mobs = np.arange(0, max_mob + 1)
        n, upb = len(g), float(g["orig_upb"].sum())
        row = {by: vintage, "mob": mobs, "pool_loans": n, "pool_upb": upb}
        for event in ("default", "payoff"):
            at = g[f"{event}_mob"].to_numpy(dtype=float)
            w = g["orig_upb"].to_numpy(dtype=float)
            hit = at[:, None] <= mobs[None, :]                         # NaN compares False: never happened
            row[f"cum_{event}_loans"] = hit.sum(axis=0)
            row[f"cum_{event}_rate_count"] = hit.sum(axis=0) / n
            row[f"cum_{event}_rate_upb"] = (hit * w[:, None]).sum(axis=0) / upb if upb > 0 else np.nan
        out.append(pd.DataFrame(row))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def checkpoint_table(curves: pd.DataFrame, by: str, metric: str = "cum_default_rate_count",
                     checkpoints=CHECKPOINTS) -> pd.DataFrame:
    """Wide table: vintage x MOB checkpoint, NaN where the vintage has not reached that MOB."""
    wide = curves[curves["mob"].isin(checkpoints)].pivot(index=by, columns="mob", values=metric)
    return wide.reindex(columns=list(checkpoints))


def vintage_checks(con: duckdb.DuckDBPyConnection, folder: Path, loans: pd.DataFrame, curves: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    panel = sql_str(folder / PANEL_FILE)
    n_loans, n_trunc = con.execute(f"SELECT count(DISTINCT loan_id), count(DISTINCT loan_id) FILTER (WHERE left_truncated) "
                                   f"FROM read_parquet({panel})").fetchone()
    pool = curves.groupby(by)["pool_loans"].first()
    add("pool loans + left-truncated loans = panel loans", int(n_loans), int(pool.sum()) + int(n_trunc), PANEL_FILE)
    decreasing = 0
    for col in ("cum_default_rate_count", "cum_payoff_rate_count", "cum_default_rate_upb", "cum_payoff_rate_upb"):
        decreasing += int((curves.groupby(by)[col].diff().fillna(0) < -1e-15).sum())
    add("cumulative curves never decrease", 0, decreasing, "vintage_curves")
    add("default + payoff never exceed the pool", 0,
        int((curves["cum_default_loans"] + curves["cum_payoff_loans"] > curves["pool_loans"]).sum()), "vintage_curves")
    # independent recount in SQL of the 12-month default rate per vintage, straight from the panel rows
    sql12 = con.execute(f"""
        WITH loan AS (
            SELECT loan_id, any_value({by}) AS v, bool_or(left_truncated) AS lt,
                   bool_or(first_bad AND mob <= 12) AS d12
            FROM read_parquet({panel}) GROUP BY loan_id)
        SELECT v, avg(CAST(d12 AS DOUBLE)) AS rate FROM loan WHERE NOT lt GROUP BY v
    """).df().set_index("v")["rate"]
    at12 = curves[curves["mob"] == 12].set_index(by)["cum_default_rate_count"]
    common = at12.index.intersection(sql12.index)
    diff = float((at12[common] - sql12[common]).abs().max()) if len(common) else 0.0
    add(f"MOB-12 default rate = SQL recount ({len(common)} vintages)", "<= 1e-12", f"{diff:.1e}", PANEL_FILE, ok=diff <= 1e-12)
    return pd.DataFrame(rows)


def vintage_markdown(curves_year: pd.DataFrame, loans: pd.DataFrame, checks: pd.DataFrame) -> str:
    n, n_trunc = len(loans), int(loans["left_truncated"].sum())
    lines = ["## Vintage curve", "",
             f"Static pool: {n - n_trunc:,} khoản có lịch sử từ đầu; {n_trunc:,} khoản bị cắt trái ({n_trunc / n:.1%}) không vào curve. "
             "Mỗi curve dừng ở MOB mà ít nhất `min_reach_share` khoản của lứa đã có thể đạt tới. Tỷ lệ tính theo số khoản; bản theo dư nợ gốc nằm trong "
             "`vintage_curves.csv`.", ""]
    for metric, title in (("cum_default_rate_count", "Default cộng dồn (lần đầu 90+ hoặc thanh lý)"),
                          ("cum_payoff_rate_count", "Trả trước cộng dồn")):
        wide = checkpoint_table(curves_year, "vintage_year", metric)
        lines += [f"### {title}", "", "| Lứa | Pool | " + " | ".join(f"MOB {k}" for k in wide.columns) + " |",
                  "|---|---:|" + "---:|" * len(wide.columns)]
        pools = curves_year.groupby("vintage_year")["pool_loans"].first()
        for v, r in wide.iterrows():
            lines.append(f"| {v} | {int(pools[v]):,} | " + " | ".join("-" if pd.isna(x) else f"{x:.2%}" for x in r) + " |")
        lines.append("")
    lines.append(f"Kiểm tra: {len(checks)}; lệch: {int((~checks['ok']).sum())}.")
    return "\n".join(lines) + "\n"
