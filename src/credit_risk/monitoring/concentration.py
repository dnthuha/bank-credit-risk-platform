"""Stage 8.4: concentration and segment view of the active book.

For each dimension in ews.yaml -> monitoring.concentration_dimensions and for
every month from twelve months before the reporting month up to it, the
active loans at the end of the month (exit_type IS NULL) are grouped:

    share        loans and UPB of the group over the active book
    delinquency  30+ and 90+ rates of the group, over its loans with a known
                 status, by count and by UPB
    trend        the group's 90+ rate pooled over the twelve previous months

Per dimension, for the reporting month:

    HHI          sum of squared UPB shares (1 / number of groups if even;
                 1.0 if everything sits in one group)
    top share    UPB share of the largest group
    mix PSI      PSI of the UPB mix against the same month a year earlier,
                 the groups as bins (thresholds from definitions.metric_thresholds.psi)

"vintage" is read as the vintage year: 56 quarters would bury the view.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_ident, sql_str
from credit_risk.monitoring.panel import MISSING, PANEL_FILE
from credit_risk.validation.metrics import psi_from_proportions

COLUMN = {"vintage": "vintage_year"}


def segment_table(con: duckdb.DuckDBPyConnection, folder: Path, dimensions: list[str], month: pd.Timestamp,
                  trailing_months: int = 12) -> pd.DataFrame:
    """One row per (dimension, group, month) for the months [month - trailing_months, month]."""
    panel = sql_str(folder / PANEL_FILE)
    first = (month - pd.DateOffset(months=trailing_months)).date()
    parts = []
    for dim in dimensions:
        col = sql_ident(COLUMN.get(dim, dim))
        parts.append(f"""
            SELECT '{dim}' AS dimension, CAST({col} AS VARCHAR) AS grp, period,
                   count(*) AS loans, sum(upb) AS upb,
                   count(*) FILTER (WHERE bucket IS NOT NULL) AS known_loans,
                   sum(upb) FILTER (WHERE bucket IS NOT NULL) AS known_upb,
                   count(*) FILTER (WHERE bucket IN ('dpd30', 'dpd60', 'dpd90plus')) AS dpd30plus_loans,
                   sum(upb) FILTER (WHERE bucket IN ('dpd30', 'dpd60', 'dpd90plus')) AS dpd30plus_upb,
                   count(*) FILTER (WHERE bucket = 'dpd90plus') AS dpd90plus_loans,
                   sum(upb) FILTER (WHERE bucket = 'dpd90plus') AS dpd90plus_upb
            FROM read_parquet({panel})
            WHERE exit_type IS NULL AND period BETWEEN ? AND ?
            GROUP BY ALL""")
    params = [first, month.date()] * len(dimensions)
    df = con.execute(" UNION ALL ".join(parts), params).df()
    df["period"] = pd.to_datetime(df["period"])
    upb_cols = [c for c in df.columns if c.endswith("_upb") or c == "upb"]
    df[upb_cols] = df[upb_cols].fillna(0.0)
    return df.sort_values(["dimension", "period", "grp"]).reset_index(drop=True)


def _rate(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def reporting_view(table: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Groups of the reporting month with shares, rates and the trailing pooled 90+ rate."""
    now = table[table["period"] == month].copy()
    total = now.groupby("dimension")[["loans", "upb"]].transform("sum")
    now["share_loans"] = now["loans"] / total["loans"]
    now["share_upb"] = now["upb"] / total["upb"]
    for w, den in (("loans", "known_loans"), ("upb", "known_upb")):
        suffix = "count" if w == "loans" else "upb"
        now[f"dpd30plus_rate_{suffix}"] = _rate(now[f"dpd30plus_{w}"], now[den])
        now[f"dpd90plus_rate_{suffix}"] = _rate(now[f"dpd90plus_{w}"], now[den])
    before = table[table["period"] < month].groupby(["dimension", "grp"])[["dpd90plus_loans", "known_loans"]].sum()
    before["dpd90plus_rate_count_trailing"] = _rate(before["dpd90plus_loans"], before["known_loans"])
    now = now.merge(before[["dpd90plus_rate_count_trailing"]], left_on=["dimension", "grp"], right_index=True, how="left")
    return now.reset_index(drop=True)


def concentration_summary(table: pd.DataFrame, month: pd.Timestamp, epsilon: float) -> pd.DataFrame:
    year_ago = month - pd.DateOffset(months=12)
    rows = []
    for dim, g in table.groupby("dimension", sort=False):
        now = g[g["period"] == month].set_index("grp")["upb"]
        share = now / now.sum()
        then = g[g["period"] == year_ago].set_index("grp")["upb"]
        groups = share.index.union(then.index)
        mix_psi = np.nan
        if len(then):
            mix_psi = psi_from_proportions((then / then.sum()).reindex(groups, fill_value=0).to_numpy(),
                                           share.reindex(groups, fill_value=0).to_numpy(), epsilon)
        rows.append({"dimension": dim, "groups": int(len(share)), "hhi_upb": float((share ** 2).sum()),
                     "top_group": str(share.idxmax()), "top_share_upb": float(share.max()), "mix_psi_vs_year_ago": mix_psi})
    return pd.DataFrame(rows)


def concentration_checks(table: pd.DataFrame, monthly: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    book = monthly.set_index("period")
    now = table[table["period"] == month]
    for dim, g in now.groupby("dimension", sort=False):
        add(f"{dim}: group loans add up to the active book ({month:%Y-%m})", int(book.loc[month, "end_loans"]), int(g["loans"].sum()),
            "portfolio_monthly")
        add(f"{dim}: group 90+ loans add up to the book's", int(book.loc[month, "dpd90plus_loans"]), int(g["dpd90plus_loans"].sum()),
            "portfolio_monthly")
    months = table["period"].nunique()
    expected = int(min(13, (book.index <= month).sum()))
    add("months covered (reporting month and up to 12 before)", expected, int(months), "segment_table")
    return pd.DataFrame(rows)


def group_order(bands: dict[str, list]) -> dict[str, list[str]]:
    """Display order per dimension: the configured band order, then "missing"; other dimensions sort as text."""
    return {f"{name}_band": [b.label for b in spec] + [MISSING] for name, spec in bands.items()}


def concentration_markdown(view: pd.DataFrame, summary: pd.DataFrame, month: pd.Timestamp, checks: pd.DataFrame,
                           psi_stable: float, psi_significant: float, order: dict[str, list[str]] | None = None,
                           max_rows: int = 12) -> str:
    order = order or {}
    def light(x):
        return "-" if pd.isna(x) else ("xanh" if x < psi_stable else "vàng" if x <= psi_significant else "đỏ")

    lines = [f"## Tập trung và phân khúc ({month:%Y-%m})", "",
             "| Chiều | Số nhóm | HHI (dư nợ) | Nhóm lớn nhất | Tỷ trọng | PSI cơ cấu so với 12 tháng trước |",
             "|---|---:|---:|---|---:|---|"]
    for r in summary.itertuples(index=False):
        psi = "-" if pd.isna(r.mix_psi_vs_year_ago) else f"{r.mix_psi_vs_year_ago:.4f} ({light(r.mix_psi_vs_year_ago)})"
        lines.append(f"| {r.dimension} | {r.groups} | {r.hhi_upb:.4f} | {r.top_group} | {r.top_share_upb:.1%} | {psi} |")
    for dim, g in view.groupby("dimension", sort=False):
        g = g.sort_values("share_upb", ascending=False)
        shown = g if dim != "property_state" else g.head(max_rows)
        lines += ["", f"### {dim}" + (f" ({max_rows} nhóm lớn nhất / {len(g)})" if len(shown) < len(g) else ""), "",
                  "| Nhóm | Tỷ trọng số khoản | Tỷ trọng dư nợ | 30+ | 90+ | 90+ (dư nợ) | 90+ gộp 12 tháng trước |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        if dim == "property_state":
            rows_shown = shown
        elif dim in order:
            rank = {g: i for i, g in enumerate(order[dim])}
            rows_shown = shown.sort_values("grp", key=lambda s: s.map(lambda g: rank.get(g, len(rank))))
        else:
            rows_shown = shown.sort_values("grp")
        for r in rows_shown.itertuples(index=False):
            lines.append(f"| {r.grp} | {r.share_loans:.1%} | {r.share_upb:.1%} | {r.dpd30plus_rate_count:.2%} | {r.dpd90plus_rate_count:.2%} | "
                         f"{r.dpd90plus_rate_upb:.2%} | {'-' if pd.isna(r.dpd90plus_rate_count_trailing) else f'{r.dpd90plus_rate_count_trailing:.2%}'} |")
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {int((~checks['ok']).sum())}."]
    return "\n".join(lines) + "\n"
