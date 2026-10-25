"""Stage 8.5: monitoring alerts, from the Stage 8.1-8.4 tables and ews.yaml rules.

Four kinds of rule, each giving one row per item with a light:

    metric       a portfolio or roll-rate figure of the reporting month against
                 the mean of the previous `baseline_trailing_months` months;
                 relative change in the "worse" direction > amber / red
    segment      a group's 90+ rate (count) against its own pooled rate of the
                 previous months; only groups with at least `min_loans` loans
    mix_psi      the UPB mix of a dimension against a year earlier, lights from
                 definitions.metric_thresholds.psi; excluded dimensions are
                 listed as excluded, not silently dropped
    new_vintage  the cumulative default rate (count) at MOB k of the most recent
                 vintage that reached k, against the mean of benchmark vintages

The overall status is the worst light. Every number a light rests on is in
the row (value, baseline, relative change, threshold), so the report and
monitoring_metrics.json can show why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LIGHTS = ("green", "amber", "red")
TEXT = {"green": "xanh", "amber": "vàng", "red": "đỏ", "excluded": "không xét"}


@dataclass(frozen=True)
class AlertSpec:
    trailing: int
    amber: float
    red: float
    metrics: list[dict]
    segment_dimensions: list[str]
    segment_min_loans: int
    mix_exclude: list[str]
    psi_stable: float
    psi_significant: float
    vintage_mobs: list[int]
    benchmark_vintages: list[int]
    limitations: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, ews: dict, definitions: dict) -> "AlertSpec":
        mon = ews["monitoring"]
        a = mon["alerts"]
        if not 0 < float(a["amber_relative_increase"]) < float(a["red_relative_increase"]):
            raise ValueError("ews.monitoring.alerts: need 0 < amber_relative_increase < red_relative_increase")
        for m in a.get("metrics", []):
            if m.get("worse") not in ("higher", "lower"):
                raise ValueError(f"ews.monitoring.alerts.metrics: {m.get('name')}: worse must be higher or lower")
        psi = definitions["metric_thresholds"]["psi"]
        return cls(trailing=int(a["baseline_trailing_months"]), amber=float(a["amber_relative_increase"]),
                   red=float(a["red_relative_increase"]), metrics=list(a.get("metrics", [])),
                   segment_dimensions=list(a.get("segments", {}).get("dimensions", [])),
                   segment_min_loans=int(a.get("segments", {}).get("min_loans", 0)),
                   mix_exclude=list(a.get("mix_psi", {}).get("exclude_dimensions", [])),
                   psi_stable=float(psi["stable_below"]), psi_significant=float(psi["significant_above"]),
                   vintage_mobs=[int(k) for k in a.get("new_vintage", {}).get("mobs", [])],
                   benchmark_vintages=[int(v) for v in a.get("new_vintage", {}).get("benchmark_vintages", [])],
                   limitations=[str(x) for x in (mon.get("report") or {}).get("limitations") or []])

    def light(self, value: float, baseline: float, worse: str = "higher") -> tuple[str, float]:
        if pd.isna(value) or pd.isna(baseline) or baseline <= 0:
            return "green", np.nan
        change = (value - baseline) / baseline if worse == "higher" else (baseline - value) / baseline
        return ("red" if change > self.red else "amber" if change > self.amber else "green"), change

    def threshold_text(self, worse: str = "higher") -> str:
        sign = "+" if worse == "higher" else "-"
        return f"vàng {sign}{self.amber:.0%}, đỏ {sign}{self.red:.0%}"


def _row(kind, item, value, baseline, change, light, threshold, detail, fmt="{:.4%}"):
    show = lambda x: "-" if pd.isna(x) else fmt.format(x)  # noqa: E731
    return {"kind": kind, "item": item, "value": value, "baseline": baseline, "relative_change": change, "light": light,
            "value_text": show(value), "baseline_text": show(baseline), "threshold": threshold, "detail": detail}


def metric_alerts(monthly: pd.DataFrame, rates: pd.DataFrame, month: pd.Timestamp, spec: AlertSpec) -> list[dict]:
    table = monthly.merge(rates, on="period", how="left").sort_values("period")
    before = table[table["period"] < month].tail(spec.trailing)
    now = table[table["period"] == month].iloc[0]
    out = []
    for m in spec.metrics:
        name, worse = m["name"], m["worse"]
        if name not in table.columns:
            raise ValueError(f"ews.monitoring.alerts.metrics: unknown metric {name!r}")
        baseline = float(before[name].mean()) if len(before) else np.nan
        light, change = spec.light(float(now[name]), baseline, worse)
        detail = "" if pd.isna(change) else f"{'tăng' if (change > 0) == (worse == 'higher') else 'giảm'} {abs(change):.0%} so với trung bình {len(before)} tháng trước"
        out.append(_row("metric", name, float(now[name]), baseline, change, light, spec.threshold_text(worse), detail))
    return out


def segment_alerts(view: pd.DataFrame, spec: AlertSpec) -> list[dict]:
    out = []
    rows = view[view["dimension"].isin(spec.segment_dimensions) & (view["loans"] >= spec.segment_min_loans)]
    for r in rows.sort_values(["dimension", "grp"]).itertuples(index=False):
        light, change = spec.light(r.dpd90plus_rate_count, r.dpd90plus_rate_count_trailing)
        detail = "" if pd.isna(change) else f"{int(r.loans):,} khoản; 90+ {'tăng' if change > 0 else 'giảm'} {abs(change):.0%} so với 12 tháng trước"
        out.append(_row("segment", f"{r.dimension} = {r.grp}", r.dpd90plus_rate_count, r.dpd90plus_rate_count_trailing, change,
                        light, spec.threshold_text() + f"; nhóm >= {spec.segment_min_loans:,} khoản", detail, "{:.2%}"))
    return out


def mix_alerts(summary: pd.DataFrame, spec: AlertSpec) -> list[dict]:
    out = []
    for r in summary.itertuples(index=False):
        psi = r.mix_psi_vs_year_ago
        if r.dimension in spec.mix_exclude:
            light, detail = "excluded", "cơ cấu theo chiều này đổi theo thiết kế mẫu, không dùng để cảnh báo"
        else:
            light = "green" if pd.isna(psi) or psi < spec.psi_stable else "amber" if psi <= spec.psi_significant else "red"
            detail = f"nhóm lớn nhất {r.top_group} ({r.top_share_upb:.1%} dư nợ)"
        out.append(_row("mix_psi", r.dimension, psi, np.nan, np.nan, light,
                        f"vàng >= {spec.psi_stable}, đỏ > {spec.psi_significant}", detail, "{:.4f}"))
    return out


def vintage_alerts(curves_year: pd.DataFrame, spec: AlertSpec) -> list[dict]:
    out = []
    for k in spec.vintage_mobs:
        at = curves_year[curves_year["mob"] == k].set_index("vintage_year")["cum_default_rate_count"]
        bench = at.reindex(spec.benchmark_vintages).dropna()
        candidates = at.drop(index=spec.benchmark_vintages, errors="ignore")
        if bench.empty or candidates.empty:
            continue
        latest = int(candidates.index.max())
        baseline = float(bench.mean())
        light, change = spec.light(float(at[latest]), baseline)
        detail = (f"lứa {latest} so với trung bình lứa {min(bench.index)}-{max(bench.index)}; "
                  f"{'cao' if change > 0 else 'thấp'} hơn {abs(change):.0%}")
        out.append(_row("new_vintage", f"default cộng dồn MOB {k}, lứa {latest}", float(at[latest]), baseline, change, light,
                        spec.threshold_text(), detail, "{:.2%}"))
    return out


def overall_status(alerts: pd.DataFrame) -> str:
    lights = [l for l in alerts["light"] if l in LIGHTS] if len(alerts) else []
    return max(lights, key=LIGHTS.index) if lights else "green"


def alerts_markdown(alerts: pd.DataFrame, status: str) -> str:
    counts = alerts["light"].value_counts()
    lines = ["## Trạng thái", "",
             f"**{TEXT[status].upper()}** — {int(counts.get('red', 0))} đỏ, {int(counts.get('amber', 0))} vàng, "
             f"{int(counts.get('green', 0))} xanh trên {int(alerts['light'].isin(LIGHTS).sum())} quy tắc.", ""]
    flagged = alerts[alerts["light"].isin(["red", "amber"])].copy()
    flagged["_rank"] = flagged["light"].map({"red": 0, "amber": 1})
    lines += ["### Cảnh báo", ""]
    if flagged.empty:
        lines.append("Không có.")
    else:
        lines += ["| Mức | Loại | Đối tượng | Giá trị | Mốc | Diễn giải |", "|---|---|---|---:|---:|---|"]
        for r in flagged.sort_values(["_rank", "kind", "item"]).itertuples(index=False):
            lines.append(f"| **{TEXT[r.light]}** | {r.kind} | {r.item} | {r.value_text} | {r.baseline_text} | {r.detail} |")
    lines += ["", "### Mọi quy tắc", "", "| Loại | Đối tượng | Giá trị | Mốc | Thay đổi | Ngưỡng | Kết quả |",
              "|---|---|---:|---:|---:|---|---|"]
    quiet = (alerts["kind"] == "segment") & (alerts["light"] == "green")
    for r in alerts[~quiet].itertuples(index=False):
        ch = "-" if pd.isna(r.relative_change) else f"{r.relative_change:+.0%}"
        lines.append(f"| {r.kind} | {r.item} | {r.value_text} | {r.baseline_text} | {ch} | {r.threshold} | {TEXT[r.light]} |")
    if quiet.any():
        lines.append(f"| segment | {int(quiet.sum())} nhóm khác (danh sách trong `alerts.csv`) | | | | | xanh |")
    return "\n".join(lines) + "\n"
