"""Page 5 - early warning: the watchlist and what it catches, then backtest, calibration, survival."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common import Bundle, model_scale, num, pct, section_missing


def _metric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("pr_auc", "roc_auc", "auc", "capture_top", "lift_top", "precision_top", "brier"))]


def render(bundle: Bundle) -> None:
    st.title("Cảnh báo sớm")
    st.markdown("Nhãn: lần đầu vào **90+ DPD hoặc thanh lý trong 3 tháng** sau tháng quan sát; khoản đã 90+ hoặc đã tất toán bị "
                "loại khỏi cohort. Feature tại tháng t chỉ dùng dòng ≤ t; dữ liệu chia theo thời gian (train → validation → "
                "stress test COVID → out-of-time). Ngưỡng cảnh báo chọn theo **năng lực xử lý**, không phải 0.5.")
    if section_missing("ews", bundle, "ews/ews_metrics.json"):
        return
    m = bundle.json("ews/ews_metrics.json") or {}
    p, sections = m.get("provenance", {}), m.get("sections", {})
    oot = p.get("out_of_time") or {}
    w = {**(p.get("watchlist") or {}), **(sections.get("watchlist_summary") or {})}
    sel = p.get("selected_model")
    cap_key = next((k for k in oot if k.startswith("capture_top")), None)
    lift_key = next((k for k in oot if k.startswith("lift_top")), None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mô hình cho watchlist", sel or "—",
              help=f"{p.get('selection_metric')} trên validation: " + ", ".join(f"{k} {num(v)}" for k, v in (p.get('validation_selection') or {}).items()))
    c2.metric(f"Capture out-of-time, {cap_key.replace('capture_top_', 'top ').replace('pct', '%') if cap_key else ''}",
              pct(oot.get(cap_key)) if cap_key else "—",
              help="Phần sự kiện 90+ trong 3 tháng tới nằm trong danh sách, trung bình các tháng out-of-time")
    c3.metric("Lift so với chọn ngẫu nhiên", f"{oot.get(lift_key):.1f}×" if lift_key and oot.get(lift_key) else "—")
    c4.metric("PR-AUC out-of-time", num(oot.get("pr_auc")), help=f"Tỷ lệ sự kiện cơ sở {pct(oot.get('event_rate'), 3)}")
    splits = p.get("splits") or {}
    st.caption("Đoạn thời gian: " + "; ".join(f"{k} {v[0]} → {v[1]}" for k, v in splits.items()) +
               f". {p.get('checks_total')} kiểm tra, {p.get('checks_failed')} lệch.")

    tabs = st.tabs([f"Watchlist {p.get('scoring_month')}", "Backtest theo đoạn", "Theo tháng", "Phân khúc", "Calibration", "Survival", "Mô hình", "Báo cáo"])
    with tabs[0]:
        _watchlist(bundle, w, p)
    with tabs[1]:
        _by_split(bundle, sel)
    with tabs[2]:
        _monthly(bundle, sel)
    with tabs[3]:
        _segments(bundle, sel)
    with tabs[4]:
        _calibration(bundle)
    with tabs[5]:
        _survival(bundle, sections)
    with tabs[6]:
        _models(bundle, p)
    with tabs[7]:
        report = bundle.text("ews/ews_report.md")
        if report:
            st.markdown(report)


def _watchlist(bundle: Bundle, w: dict, p: dict) -> None:
    watch = bundle.csv("ews/watchlist.csv")
    c1, c2, c3 = st.columns(3)
    c1.metric("Khoản trong danh sách", f"{w.get('listed', 0):,}", help=f"Top {pct(w.get('capacity_share'), 0)} theo PD trong {w.get('scoring_rows', 0):,} khoản của cohort")
    c2.metric("Kỳ vọng vào 90+ trong 3 tháng", f"{w.get('expected_events', 0):.0f}", help=f"PD trung bình trong danh sách {pct(w.get('mean_pd'))}")
    c3.metric("Dư nợ trong danh sách", f"{w.get('upb', 0):,.0f}" if w.get("upb") else "—")
    by_bucket = w.get("by_bucket") or {}
    if by_bucket:
        st.markdown("Trạng thái tại t: " + ", ".join(f"**{k}** {v:,}" for k, v in sorted(by_bucket.items(), key=lambda kv: -kv[1])))
    reasons = w.get("top_reasons") or {}
    if reasons:
        df = pd.DataFrame({"feature": list(reasons), "count": list(reasons.values())})
        chart = alt.Chart(df).mark_bar(color="#d9480f").encode(y=alt.Y("feature:N", sort=None, title=None), x=alt.X("count:Q", title="Số lần xuất hiện trong reason code"))
        st.altair_chart(chart.properties(height=22 * len(df), title="Yếu tố đẩy điểm lên thường gặp nhất"), use_container_width=True)
    if watch is None:
        return
    st.markdown("**Danh sách** (lọc được; tải về đầy đủ)")
    f1, f2 = st.columns(2)
    buckets = sorted(watch["bucket_at_t"].dropna().unique()) if "bucket_at_t" in watch else []
    pick_b = f1.multiselect("Trạng thái tại t", buckets, default=buckets)
    reason_cols = [c for c in watch.columns if c.startswith("reason_")]
    all_reasons = sorted(set(watch[reason_cols].stack().dropna().unique())) if reason_cols else []
    pick_r = f2.multiselect("Có reason code", all_reasons)
    shown = watch
    if pick_b and "bucket_at_t" in shown:
        shown = shown[shown["bucket_at_t"].isin(pick_b)]
    if pick_r and reason_cols:
        shown = shown[shown[reason_cols].isin(pick_r).any(axis=1)]
    fmt = {"pd": "{:.2%}"}
    st.dataframe(shown.head(500).style.format(fmt, na_rep=""), hide_index=True, use_container_width=True, height=420)
    st.caption(f"{len(shown):,} khoản sau lọc (hiển thị tối đa 500).")
    st.download_button("Tải watchlist.csv", watch.to_csv(index=False).encode("utf-8"), file_name=f"watchlist_{p.get('scoring_month')}.csv", mime="text/csv")


def _by_split(bundle: Bundle, sel: str | None) -> None:
    df = bundle.csv("ews/by_split.csv")
    if df is None:
        st.info("Không có by_split.csv.")
        return
    cols = ["model", "split", "rows", "events", "event_rate", "mean_pd", *_metric_cols(df)]
    show = df[[c for c in cols if c in df]]
    fmt = {c: "{:.3%}" for c in ("event_rate", "mean_pd") if c in show}
    fmt.update({c: "{:.1%}" for c in show.columns if c.startswith(("capture", "precision"))})
    fmt.update({c: "{:.3f}" for c in show.columns if c.startswith(("pr_auc", "roc_auc", "auc", "brier"))})
    fmt.update({c: "{:.1f}" for c in show.columns if c.startswith("lift")})
    st.dataframe(show.style.format(fmt, na_rep=""), hide_index=True, use_container_width=True)
    cap = next((c for c in df.columns if c.startswith("capture_top")), None)
    if cap:
        order = ["train", "validation", "stress_test", "out_of_time"]
        d = df[df["split"].isin(order)].copy()
        chart = alt.Chart(d).mark_bar().encode(
            x=alt.X("split:N", sort=order, title=None), y=alt.Y(f"{cap}:Q", title="Capture", axis=alt.Axis(format="%")),
            color=alt.Color("model:N", scale=model_scale(), title=None), xOffset="model:N",
            tooltip=["model", "split", alt.Tooltip(f"{cap}:Q", format=".1%")],
        ).properties(height=280, title="Capture ở công suất danh sách, theo đoạn thời gian (stress test = COVID)")
        st.altair_chart(chart, use_container_width=True)


def _monthly(bundle: Bundle, sel: str | None) -> None:
    df = bundle.csv("ews/monthly_lists.csv")
    if df is None:
        st.info("Không có monthly_lists.csv.")
        return
    if "model" in df and sel in set(df["model"]):
        df = df[df["model"] == sel]
    period_col = next((c for c in ("period", "month", "scoring_month") if c in df), None)
    cap = next((c for c in df.columns if c.startswith("capture")), None)
    if period_col and cap:
        d = df.copy()
        d["date"] = pd.to_datetime(d[period_col].astype(str).str[:7])
        base = alt.Chart(d).encode(x=alt.X("date:T", title=None))
        line = base.mark_line(point=True).encode(y=alt.Y(f"{cap}:Q", title="Capture", axis=alt.Axis(format="%")),
                                                 tooltip=[period_col, alt.Tooltip(f"{cap}:Q", format=".1%"), *[c for c in ("events", "listed", "split") if c in d]])
        if "events" in d:
            bars = base.mark_bar(opacity=0.25, color="#9e9e9e").encode(y=alt.Y("events:Q", title="Sự kiện trong tháng"))
            st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=320, title=f"Capture từng tháng ({sel}), cột xám = số sự kiện"), use_container_width=True)
        else:
            st.altair_chart(line.properties(height=320), use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True, height=300)


def _segments(bundle: Bundle, sel: str | None) -> None:
    df = bundle.csv("ews/segments.csv")
    if df is None:
        st.info("Không có segments.csv.")
        return
    splits = list(df["split"].unique()) if "split" in df else []
    pick = st.selectbox("Đoạn", splits, index=splits.index("out_of_time") if "out_of_time" in splits else 0) if splits else None
    d = df[df["split"] == pick] if pick else df
    if "model" in d and sel in set(d["model"]):
        d = d[d["model"] == sel]
    fmt = {c: "{:.1%}" for c in d.columns if c.startswith(("capture", "precision", "share"))}
    fmt.update({c: "{:.3%}" for c in ("event_rate", "mean_pd") if c in d})
    st.dataframe(d.style.format(fmt, na_rep=""), hide_index=True, use_container_width=True, height=450)
    st.caption("Khoản đang current tại t là phần khó và có giá trị nhất: khoản đã 30 / 60 ngày quá hạn gần như luôn có mặt trong danh sách.")


def _calibration(bundle: Bundle) -> None:
    df = bundle.csv("ews/calibration.csv")
    if df is None:
        st.info("Không có calibration.csv.")
        return
    st.markdown("PD trung bình so với tỷ lệ sự kiện thực tế theo band, từng đoạn. Quyết định: **không hiệu chỉnh lại**, báo cáo riêng.")
    x = next((c for c in ("mean_pd", "pd_mean") if c in df), None)
    y = next((c for c in ("event_rate", "observed_rate", "rate") if c in df), None)
    if x and y and "split" in df:
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X(f"{x}:Q", title="PD trung bình", axis=alt.Axis(format="%")), y=alt.Y(f"{y}:Q", title="Tỷ lệ sự kiện", axis=alt.Axis(format="%")),
            color=alt.Color("split:N", title=None), tooltip=list(df.columns))
        diag = alt.Chart(pd.DataFrame({"v": [0, float(df[[x, y]].max().max())]})).mark_line(strokeDash=[4, 4], color="grey").encode(x="v:Q", y="v:Q")
        st.altair_chart((chart + diag).properties(height=320), use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True, height=300)


def _survival(bundle: Bundle, sections: dict) -> None:
    df = bundle.csv("ews/survival_curves.csv")
    if df is None:
        st.info("Không có survival_curves.csv.")
        return
    st.markdown("Kaplan–Meier trả lời “nếu không ai trả trước”; Aalen–Johansen (CIF) trả lời “bao nhiêu phần danh mục thực sự default” "
                "khi trả trước là rủi ro cạnh tranh.")
    group_col = next((c for c in ("group", "segment", "level", "stratum") if c in df), None)
    dim_col = next((c for c in ("dimension", "by") if c in df), None)
    d = df
    if dim_col:
        dims = list(df[dim_col].unique())
        pick = st.selectbox("Phân theo", dims)
        d = df[df[dim_col] == pick]
    ycols = [c for c in d.columns if c.startswith(("cif", "km", "survival", "cum"))]
    ycol = st.selectbox("Đường cong", ycols) if ycols else None
    if ycol and "mob" in d:
        enc = {"x": alt.X("mob:Q", title="MOB"), "y": alt.Y(f"{ycol}:Q", title=None, axis=alt.Axis(format="%")), "tooltip": list(d.columns)}
        if group_col:
            enc["color"] = alt.Color(f"{group_col}:N", title=None)
        st.altair_chart(alt.Chart(d).mark_line().encode(**enc).properties(height=340), use_container_width=True)
    cps = sections.get("survival_checkpoints")
    if cps:
        with st.expander("Mốc MOB"):
            st.dataframe(pd.DataFrame(cps), hide_index=True, use_container_width=True)


def _models(bundle: Bundle, p: dict) -> None:
    imp = bundle.csv("ews/importance.csv")
    coef = bundle.csv("ews/coefficients.csv")
    trials = bundle.csv("ews/trials.csv")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**LightGBM challenger: biến quan trọng**")
        if imp is not None:
            val = next((c for c in ("mean_abs_shap", "gain_share", "gain", "importance") if c in imp), None)
            top = imp.head(20)
            if val:
                st.altair_chart(alt.Chart(top).mark_bar(color="#d9480f").encode(y=alt.Y("feature:N", sort=None, title=None), x=alt.X(f"{val}:Q", title=val))
                                .properties(height=22 * len(top)), use_container_width=True)
            else:
                st.dataframe(top, hide_index=True)
    with c2:
        st.markdown("**Logistic baseline: hệ số**")
        if coef is not None:
            st.dataframe(coef.head(40), hide_index=True, use_container_width=True, height=460)
    if trials is not None:
        with st.expander(f"Các cấu hình đã thử ({len(trials)})"):
            st.dataframe(trials, hide_index=True, use_container_width=True)
    fp = p.get("fingerprints") or {}
    st.caption("Fingerprint: " + ", ".join(f"{k} `{str(v)[:12]}`" for k, v in fp.items()) + f"; best iteration {p.get('challenger_best_iteration')}.")
