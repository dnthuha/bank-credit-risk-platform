"""Stage 8.1: portfolio metrics for every reporting month, by count and by UPB.

All from the monitoring panel (Stage 8.0), in one DuckDB aggregation:

    start of month m    loans active at the end of m-1 (has_prev and the previous
                        row has no exit); their UPB is prev_upb
    end of month m      rows of m with no exit; their UPB is upb
    delinquency         dpd30 / dpd60 / dpd90plus / 30+ shares of the loans active
                        at the end of m whose status is known (unknown status is
                        counted, not guessed)
    new defaults        first month a loan is 90+ or liquidated (first_bad), over
                        the start of month; a re-default after cure is not new
    payoff (SMM)        voluntary payoffs over the start of month; CPR =
                        1 - (1 - SMM)^12. Full payoffs only: the panel has no
                        partial prepayments, so this is a lower bound of the
                        usual SMM
    liquidations, other exits, entries (first row of a loan, or the first row
                        after a reporting gap)

UPB-weighted rates weigh each loan by its UPB at the point the rate is
measured: prev_upb for flows out of the start of month (defaults, payoffs),
upb for shares of the end of month (delinquency).

Every month is tagged with its regime from definitions.covid_period:
pre_covid / covid (stress segment) / post_covid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from credit_risk.data.db import sql_str
from credit_risk.monitoring.panel import PANEL_FILE

DELINQUENCY = ("dpd30", "dpd60", "dpd90plus")


@dataclass(frozen=True)
class PortfolioSpec:
    covid_start: pd.Timestamp
    covid_end: pd.Timestamp
    reporting_month: pd.Timestamp | None
    codes: dict[str, Any]
    zero_balance: dict[str, list[str]]

    @classmethod
    def from_config(cls, ews: dict, definitions: dict) -> "PortfolioSpec":
        covid = definitions["covid_period"]
        month = ews["monitoring"].get("reporting_month")
        return cls(covid_start=pd.Timestamp(f"{covid['start']}-01"), covid_end=pd.Timestamp(f"{covid['end']}-01"),
                   reporting_month=pd.Timestamp(f"{month}-01") if month else None,
                   codes=definitions["mortgage"]["delinquency_status_codes"],
                   zero_balance=definitions["mortgage"]["zero_balance_codes"])

    def regime(self, period: pd.Series) -> pd.Series:
        p = pd.to_datetime(period)
        return pd.Series(np.select([p < self.covid_start, p <= self.covid_end], ["pre_covid", "covid"], "post_covid"),
                         index=period.index)


def monthly_sql(panel: Path) -> str:
    start = "has_prev AND prev_exit_type IS NULL"
    end = "exit_type IS NULL"
    known = f"{end} AND bucket IS NOT NULL"
    parts = [
        f"count(*) FILTER (WHERE {start}) AS start_loans",
        f"sum(prev_upb) FILTER (WHERE {start}) AS start_upb",
        f"count(*) FILTER (WHERE {end}) AS end_loans",
        f"sum(upb) FILTER (WHERE {end}) AS end_upb",
        f"count(*) FILTER (WHERE {end} AND bucket IS NULL) AS end_unknown_status",
        f"count(*) FILTER (WHERE {known}) AS known_loans",
        f"sum(upb) FILTER (WHERE {known}) AS known_upb",
    ]
    for b in DELINQUENCY:
        parts += [f"count(*) FILTER (WHERE {known} AND bucket = '{b}') AS {b}_loans",
                  f"sum(upb) FILTER (WHERE {known} AND bucket = '{b}') AS {b}_upb"]
    for name, cond in (("new_default", "first_bad"), ("payoff", "exit_type = 'payoff'"),
                       ("liquidation", "exit_type = 'liquidation'"), ("other_exit", "exit_type = 'other_exit'")):
        parts += [f"count(*) FILTER (WHERE {start} AND {cond}) AS {name}_loans",
                  f"sum(prev_upb) FILTER (WHERE {start} AND {cond}) AS {name}_upb",
                  f"count(*) FILTER (WHERE NOT ({start}) AND {cond}) AS {name}_outside_start"]
    parts += ["count(*) FILTER (WHERE NOT has_prev) AS entries",
              "count(*) FILTER (WHERE NOT has_prev AND exit_type IS NULL) AS entries_active",
              f"count(*) FILTER (WHERE {start} AND exit_type IS NOT NULL) AS exits_from_start"]
    return f"SELECT period, {', '.join(parts)} FROM read_parquet({sql_str(panel)}) GROUP BY period ORDER BY period"


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num, den = num.astype(float), den.astype(float)
    return (num / den.where(den > 0)).astype(float)


def portfolio_monthly(con: duckdb.DuckDBPyConnection, folder: Path, spec: PortfolioSpec) -> pd.DataFrame:
    df = con.execute(monthly_sql(folder / PANEL_FILE)).df()
    df["period"] = pd.to_datetime(df["period"])
    upb_cols = [c for c in df.columns if c.endswith("_upb")]
    df[upb_cols] = df[upb_cols].fillna(0.0)
    df.insert(1, "regime", spec.regime(df["period"]))
    for b in DELINQUENCY:
        df[f"{b}_rate_count"] = _ratio(df[f"{b}_loans"], df["known_loans"])
        df[f"{b}_rate_upb"] = _ratio(df[f"{b}_upb"], df["known_upb"])
    df["dpd30plus_rate_count"] = df[[f"{b}_rate_count" for b in DELINQUENCY]].sum(axis=1, min_count=1)
    df["dpd30plus_rate_upb"] = df[[f"{b}_rate_upb" for b in DELINQUENCY]].sum(axis=1, min_count=1)
    for name in ("new_default", "payoff", "liquidation", "other_exit"):
        df[f"{name}_rate_count"] = _ratio(df[f"{name}_loans"], df["start_loans"])
        df[f"{name}_rate_upb"] = _ratio(df[f"{name}_upb"], df["start_upb"])
    for w in ("count", "upb"):
        df[f"cpr_{w}"] = 1 - (1 - df[f"payoff_rate_{w}"]) ** 12
    return df


def reporting_month(monthly: pd.DataFrame, spec: PortfolioSpec) -> pd.Timestamp:
    months = pd.to_datetime(monthly["period"])
    if spec.reporting_month is None:
        return months.max()
    if spec.reporting_month not in set(months):
        raise ValueError(f"ews.monitoring.reporting_month {spec.reporting_month:%Y-%m} is not in the panel "
                         f"({months.min():%Y-%m}..{months.max():%Y-%m})")
    return spec.reporting_month


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def portfolio_checks(con: duckdb.DuckDBPyConnection, folder: Path, monthly: pd.DataFrame,
                     spec: PortfolioSpec, month: pd.Timestamp) -> pd.DataFrame:
    rows = []

    def add(check, claimed, recounted, source, ok=None):
        rows.append({"check": check, "claimed": claimed, "recounted": recounted, "source": source,
                     "ok": bool(claimed == recounted) if ok is None else bool(ok)})

    perf = sql_str(folder / "performance.parquet")
    # exits counted from the raw performance file, not from the panel
    zb = spec.zero_balance
    raw = con.execute(f"""
        SELECT count(*) FILTER (WHERE trim(zero_balance_code) IN ({', '.join(sql_str(c) for c in zb['voluntary_payoff'])})) AS payoff,
               count(*) FILTER (WHERE trim(zero_balance_code) IN ({', '.join(sql_str(c) for c in zb['liquidation'])})) AS liquidation
        FROM read_parquet({perf})
    """).fetchone()
    for name, expected in zip(("payoff", "liquidation"), raw):
        got = int(monthly[f"{name}_loans"].sum() + monthly[f"{name}_outside_start"].sum())
        add(f"{name} exits over all months = raw zero_balance_code count", int(expected), got, "performance.parquet")

    # the reporting month's 90+ count straight from the raw status codes
    codes = spec.codes
    raw90 = con.execute(f"""
        SELECT count(*) FROM read_parquet({perf})
        WHERE period = ? AND nullif(trim(zero_balance_code), '') IS NULL
          AND (TRY_CAST(trim(delinquency_status) AS INTEGER) >= {int(codes['dpd90plus_min_numeric_code'])}
               OR trim(delinquency_status) = {sql_str(codes['reo_acquisition'])})
    """, [month.date()]).fetchone()[0]
    row = monthly[monthly["period"] == month].iloc[0]
    add(f"{month:%Y-%m} dpd90plus loans = raw status count", int(raw90), int(row["dpd90plus_loans"]), "performance.parquet")

    # roll-forward: end(m) = start(m) - exits from start(m) + active entries(m)
    rf = monthly["start_loans"] - monthly["exits_from_start"] + monthly["entries_active"]
    add("roll-forward end = start - exits + entries, every month", 0, int((rf != monthly["end_loans"]).sum()), "panel")
    # loans that vanish without an exit row (reporting gaps), from end(m-1) to start(m)
    vanished = (monthly["end_loans"].shift(1) - monthly["start_loans"]).iloc[1:]
    add("loans leaving without an exit row (gaps): never negative", ">= 0 every month", int(vanished.sum()),
        "panel", ok=int((vanished < 0).sum()) == 0)

    rates = monthly[[c for c in monthly.columns if "_rate_" in c]]
    add("every rate within [0, 1]", 0, int(((rates < 0) | (rates > 1)).sum().sum()), "portfolio_monthly")
    share = monthly[[f"{b}_loans" for b in DELINQUENCY]].sum(axis=1) <= monthly["known_loans"]
    add("delinquent loans <= loans with known status, every month", 0, int((~share).sum()), "portfolio_monthly")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------------

HEADLINE = [
    ("end_loans", "Khoản đang hoạt động", "{:,.0f}"),
    ("end_upb", "Dư nợ đang hoạt động (tỷ USD)", "{:,.2f}"),
    ("dpd30plus_rate_count", "30+ (số khoản)", "{:.3%}"),
    ("dpd30plus_rate_upb", "30+ (dư nợ)", "{:.3%}"),
    ("dpd90plus_rate_count", "90+ (số khoản)", "{:.3%}"),
    ("dpd90plus_rate_upb", "90+ (dư nợ)", "{:.3%}"),
    ("new_default_rate_count", "Default mới / tháng (số khoản)", "{:.4%}"),
    ("new_default_rate_upb", "Default mới / tháng (dư nợ)", "{:.4%}"),
    ("cpr_count", "CPR (số khoản)", "{:.2%}"),
    ("cpr_upb", "CPR (dư nợ)", "{:.2%}"),
]


def regime_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Mean of each headline rate by regime (simple mean over months)."""
    cols = [c for c, _, _ in HEADLINE if "rate" in c or c.startswith("cpr")]
    return monthly.groupby("regime", sort=False)[cols].mean().reset_index()


def portfolio_markdown(monthly: pd.DataFrame, month: pd.Timestamp, checks: pd.DataFrame) -> str:
    row = monthly[monthly["period"] == month].iloc[0]
    prev = monthly[monthly["period"] < month].tail(12)
    lines = [f"## Danh mục tháng {month:%Y-%m}", "",
             "| Chỉ số | Tháng báo cáo | Trung bình 12 tháng trước |", "|---|---:|---:|"]
    for col, label, fmt in HEADLINE:
        scale = 1e9 if col == "end_upb" else 1
        lines.append(f"| {label} | {fmt.format(row[col] / scale)} | {fmt.format(prev[col].mean() / scale) if len(prev) else '-'} |")
    lines += ["", f"Trạng thái không rõ cuối tháng (mã XX): {int(row['end_unknown_status']):,} khoản, không tính vào mẫu số tỷ lệ quá hạn.",
              "", "### Theo giai đoạn (trung bình các tháng)", "",
              "| Giai đoạn | Số tháng | 30+ (số khoản) | 90+ (số khoản) | 90+ (dư nợ) | Default mới / tháng | CPR (số khoản) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for regime, part in monthly.groupby("regime", sort=False):
        lines.append(f"| {regime} | {len(part)} | {part['dpd30plus_rate_count'].mean():.3%} | {part['dpd90plus_rate_count'].mean():.3%} | "
                     f"{part['dpd90plus_rate_upb'].mean():.3%} | {part['new_default_rate_count'].mean():.4%} | {part['cpr_count'].mean():.2%} |")
    lines += ["", f"Kiểm tra: {len(checks)}; lệch: {int((~checks['ok']).sum())}."]
    return "\n".join(lines) + "\n"
