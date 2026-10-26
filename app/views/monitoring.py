"""Page 4 - portfolio monitoring: status and alerts, then the series that explain them."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common import LIGHT_COLOURS, Bundle, badge, light_label, light_scale, pct, section_missing

REGIME_COLOURS = {"pre_covid": "#e3f2fd", "covid": "#fff3e0", "post_covid": "#e8f5e9"}


def render(bundle: Bundle) -> None:
    st.title("Giám sát danh mục")
    st.markdown("Freddie Mac sample, một dòng là một cặp khoản vay – tháng. Chỉ số theo số khoản và dư nợ, vintage curve "
                "(so các lứa ở cùng MOB), roll rate và ma trận dịch chuyển, tập trung rủi ro. Đổi tháng báo cáo trong "
                "`ews.yaml` là mọi bảng tự làm mới.")
    if section_missing("monitoring", bundle, "monitoring/monitoring_metrics.json"):
        return
    m = bundle.json("monitoring/monitoring_metrics.json") or {}
    p = m.get("provenance", {})
    month = p.get("reporting_month")
    status = m.get("status")
    alerts = pd.DataFrame(m.get("alerts") or [])

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"### {month}")
    c1.markdown(badge(f"Trạng thái: {light_label(status)}", LIGHT_COLOURS.get(status, "#607d8b")), unsafe_allow_html=True)
    monthly = bundle.csv("monitoring/portfolio_monthly.csv")
    row = None
    if monthly is not None:
        monthly["period"] = monthly["period"].astype(str).str[:7]
        last = monthly[monthly["period"] == month]
        row = last.iloc[0] if len(last) else None
    if row is not None:
        c2.metric("30+ DPD (khoản)", pct(row.get("dpd30plus_rate_count"), 3))
        c3.metric("90+ DPD (khoản)", pct(row.get("dpd90plus_rate_count"), 3))
        c4.metric("90+ DPD (dư nợ)", pct(row.get("dpd90plus_rate_upb"), 3))
    panel = p.get("panel") or {}
    st.caption(f"Panel {panel.get('rows', 0):,} dòng, {panel.get('loans', 0):,} khoản, {panel.get('first_month')} → "
               f"{panel.get('last_month')}; {p.get('checks_total')} kiểm tra, {p.get('checks_failed')} lệch.")

    if len(alerts):
        flagged = alerts[alerts["light"].isin(["red", "amber"])].sort_values("light")
        st.markdown("**Cảnh báo vàng / đỏ** (so với trung bình 12 tháng trước, hoặc lứa so với lứa ở cùng MOB)")
        if len(flagged):
            for r in flagged.itertuples(index=False):
                st.markdown(f"{badge(light_label(r.light), LIGHT_COLOURS[r.light])} **{r.kind} · {r.item}**: "
                            f"{r.value_text} so với {r.baseline_text} — {r.detail} (ngưỡng {r.threshold})", unsafe_allow_html=True)
        else:
            st.success("Không có cảnh báo vàng / đỏ.")
        with st.expander(f"Mọi quy tắc đã kiểm ({len(alerts)})"):
            st.dataframe(alerts, hide_index=True, use_container_width=True)

    tabs = st.tabs(["Chuỗi tháng", "Vintage", "Roll rate và migration", "Phân khúc và tập trung", "Báo cáo"])
    with tabs[0]:
        _monthly(monthly)
    with tabs[1]:
        _vintage(bundle, m)
    with tabs[2]:
        _roll(bundle, m)
    with tabs[3]:
        _segments(bundle, m)
    with tabs[4]:
        report = bundle.text("monitoring/monitoring_report.md")
        if report:
            st.markdown(report)


def _regimes(monthly: pd.DataFrame) -> alt.Chart:
    """Background bands for pre_covid / covid / post_covid."""
    r = monthly[["period", "regime"]].copy()
    r["date"] = pd.to_datetime(r["period"])
    blocks = []
    for regime, part in r.groupby("regime", sort=False):
        blocks.append({"regime": regime, "start": part["date"].min(), "end": part["date"].max() + pd.offsets.MonthEnd(0)})
    b = pd.DataFrame(blocks)
    return alt.Chart(b).mark_rect(opacity=0.5).encode(
        x="start:T", x2="end:T", color=alt.Color("regime:N", scale=alt.Scale(domain=list(REGIME_COLOURS), range=list(REGIME_COLOURS.values())),
                                                 legend=alt.Legend(title="Giai đoạn")))


def _monthly(monthly: pd.DataFrame | None) -> None:
    if monthly is None:
        st.info("Không có portfolio_monthly.csv.")
        return
    basis = st.radio("Cơ sở", ["count", "upb"], horizontal=True, format_func=lambda b: "Theo số khoản" if b == "count" else "Theo dư nợ")
    cols = [c for c in (f"dpd30plus_rate_{basis}", f"dpd60plus_rate_{basis}", f"dpd90plus_rate_{basis}", f"new_default_rate_{basis}")
            if c in monthly]
    if not cols:
        st.info("Thiếu cột tỷ lệ.")
        return
    df = monthly[["period", "regime", *cols]].copy()
    df["date"] = pd.to_datetime(df["period"])
    long = df.melt(id_vars=["date", "period", "regime"], value_vars=cols, var_name="series", value_name="rate")
    long["series"] = long["series"].str.replace(f"_rate_{basis}", "", regex=False)
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None), y=alt.Y("rate:Q", title="Tỷ lệ", axis=alt.Axis(format=".2%")),
        color=alt.Color("series:N", title=None), tooltip=["period", "regime", "series", alt.Tooltip("rate:Q", format=".3%")])
    st.altair_chart((_regimes(monthly) + lines).resolve_scale(color="independent").properties(height=320,
                    title="Tỷ lệ quá hạn và default mới theo tháng (nền: giai đoạn COVID)"), use_container_width=True)
    size_cols = [c for c in ("end_loans", "end_upb", "entries", "payoff_loans", "new_default_loans") if c in monthly]
    with st.expander("Quy mô danh mục theo tháng"):
        st.dataframe(monthly[["period", "regime", *size_cols]].iloc[::-1], hide_index=True, use_container_width=True, height=350)


def _vintage(bundle: Bundle, m: dict) -> None:
    curves = bundle.csv("monitoring/vintage_curves_year.csv")
    if curves is None:
        st.info("Không có vintage_curves_year.csv.")
        return
    metric = st.selectbox("Chỉ số", ["cum_default_rate_count", "cum_default_rate_upb", "cum_payoff_rate_count"],
                          format_func=lambda c: {"cum_default_rate_count": "Default tích luỹ (khoản)", "cum_default_rate_upb": "Default tích luỹ (dư nợ)",
                                                 "cum_payoff_rate_count": "Trả hết tích luỹ (khoản)"}[c])
    df = curves.dropna(subset=[metric]).copy()
    df["vintage_year"] = df["vintage_year"].astype(int).astype(str)
    max_mob = st.slider("MOB tối đa", 12, int(df["mob"].max()), min(60, int(df["mob"].max())), step=12)
    df = df[df["mob"] <= max_mob]
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("mob:Q", title="Tháng kể từ giải ngân (MOB)"), y=alt.Y(f"{metric}:Q", title=None, axis=alt.Axis(format=".2%")),
        color=alt.Color("vintage_year:N", title="Lứa", scale=alt.Scale(scheme="turbo")),
        tooltip=["vintage_year", "mob", "pool_loans", alt.Tooltip(f"{metric}:Q", format=".3%")],
    ).properties(height=380, title="Vintage curve: so các lứa ở cùng MOB mới có nghĩa")
    st.altair_chart(chart, use_container_width=True)
    cp = (m.get("sections") or {}).get("vintage_checkpoints") or {}
    if cp.get(metric):
        st.markdown("**Mốc MOB theo lứa**")
        st.dataframe(pd.DataFrame(cp[metric]), hide_index=True, use_container_width=True)


def _roll(bundle: Bundle, m: dict) -> None:
    roll = bundle.csv("monitoring/roll_rates.csv")
    if roll is not None:
        roll = roll.copy()
        roll["date"] = pd.to_datetime(roll["period"])
        cols = [c for c in roll.columns if c.endswith("_count") and "_to_" in c]
        pick = st.multiselect("Chuyển nhóm", cols, default=[c for c in cols if c in ("current_to_dpd30_count", "dpd30_to_dpd60_count", "dpd60_to_dpd90plus_count")])
        if pick:
            long = roll.melt(id_vars=["date", "period"], value_vars=pick, var_name="transition", value_name="rate").dropna()
            long["transition"] = long["transition"].str.replace("_count", "", regex=False)
            chart = alt.Chart(long).mark_line().encode(
                x=alt.X("date:T", title=None), y=alt.Y("rate:Q", title="Xác suất chuyển sau một tháng", axis=alt.Axis(format=".1%")),
                color=alt.Color("transition:N", title=None), tooltip=["period", "transition", alt.Tooltip("rate:Q", format=".2%")],
            ).properties(height=320, title="Roll rate theo tháng (chỉ báo nhạy nhất về sức khoẻ danh mục)")
            st.altair_chart(chart, use_container_width=True)
    mig = (m.get("sections") or {}).get("migration") or {}
    if mig:
        key = st.selectbox("Ma trận dịch chuyển", list(mig), format_func=lambda k: {"reporting_month": f"Tháng báo cáo {m.get('provenance', {}).get('reporting_month')}"}.get(k, k))
        mat = pd.DataFrame(mig[key])
        if len(mat):
            first = mat.columns[0]
            st.dataframe(mat.set_index(first).style.format("{:.2%}"),
                         use_container_width=True)
            st.caption("Chuẩn hoá theo dòng: mỗi dòng cộng lại bằng 1 (từ trạng thái dòng sang trạng thái cột).")


def _segments(bundle: Bundle, m: dict) -> None:
    conc = bundle.csv("monitoring/concentration.csv")
    if conc is not None:
        st.markdown("**Tập trung rủi ro** (HHI theo dư nợ, nhóm lớn nhất, PSI cơ cấu so với một năm trước)")
        st.dataframe(conc.style.format({"hhi_upb": "{:.3f}", "top_share_upb": "{:.1%}", "mix_psi_vs_year_ago": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
    seg = (m.get("sections") or {}).get("segments_reporting")
    if seg:
        df = pd.DataFrame(seg)
        st.markdown("**Phân khúc ở tháng báo cáo**")
        fmt = {c: "{:.2%}" for c in df.columns if "rate" in c or "share" in c}
        st.dataframe(df.style.format(fmt, na_rep=""), hide_index=True, use_container_width=True, height=400)
