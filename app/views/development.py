"""Page 2 - model development: the scorecard, the challenger, the calibration, and one applicant explained."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common import Bundle, markdown_section, model_scale, num, pct, section_missing


def render(bundle: Bundle) -> None:
    st.title("Xây dựng mô hình")
    st.markdown("Home Credit: một dòng là một hồ sơ tại ngày nộp đơn. Champion là scorecard logistic trên WoE "
                "(giải thích được từng điểm), challenger là LightGBM. Cả hai dùng đúng một tập dân số và một split.")
    if section_missing("development", bundle, "development/performance.json"):
        return

    tab_app, tab_card, tab_ch, tab_cal = st.tabs(["Một hồ sơ cụ thể", "Scorecard", "Challenger LightGBM", "Calibration"])
    with tab_app:
        _applicant(bundle)
    with tab_card:
        _scorecard(bundle)
    with tab_ch:
        _challenger(bundle)
    with tab_cal:
        _calibration(bundle)


# ---------------------------------------------------------------------------

def _applicant(bundle: Bundle) -> None:
    ex = bundle.json("development/applicant_examples.json")
    if not ex:
        st.info("Bundle không có `applicant_examples.json` (cần gói model và features.parquet khi chạy `report`).")
        return
    apps = ex["applicants"]
    labels = {a["SK_ID_CURR"]: f"#{a['SK_ID_CURR']} · score {a['score']} · PD {pct(a['pd_champion'])}"
              + (" · bad" if a.get("target") == 1 else "") for a in apps}
    chosen = st.selectbox("Hồ sơ (mẫu test, xếp từ rủi ro nhất)", list(labels), format_func=labels.get)
    a = next(x for x in apps if x["SK_ID_CURR"] == chosen)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", a["score"], delta=f"{a['score'] - a['base_points']:+d} so với điểm cơ sở {a['base_points']}")
    c2.metric("PD scorecard (đã calibrate)", pct(a["pd_champion"], 2))
    c3.metric("PD LightGBM (đã calibrate)", pct(a["pd_challenger"], 2))
    c4.metric("Kết cục thật", {1: "Bad", 0: "Good"}.get(a.get("target"), "—"))

    st.markdown("**Mã lý do** (scorecard, các biến làm mất nhiều điểm nhất so với bin tốt nhất): "
                + ", ".join(f"`{r}`" for r in a["reason_codes"]))
    st.caption("LightGBM (SHAP): " + ", ".join(f"`{r}`" for r in a["reason_codes_challenger"]))

    pts = pd.DataFrame(a["points"]).sort_values("lost", ascending=False)
    pts["label"] = pts["feature"] + " = " + pts["bin"].astype(str)
    top = pts.head(15)
    chart = alt.Chart(top).mark_bar().encode(
        y=alt.Y("label:N", sort=None, title=None),
        x=alt.X("points:Q", title="Điểm của biến (âm = trừ điểm)"),
        color=alt.condition(alt.datum.points < 0, alt.value("#c62828"), alt.value("#2e7d32")),
        tooltip=["feature", "value", "bin", "points", "best_points", "lost"],
    ).properties(height=28 * len(top), title="15 biến mất nhiều điểm nhất so với bin tốt nhất của nó")
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Toàn bộ điểm theo biến và thông tin hồ sơ"):
        st.dataframe(pts[["feature", "value", "bin", "points", "best_points", "lost"]], hide_index=True,
                     use_container_width=True)
        st.json(a["descriptive"])


def _scorecard(bundle: Bundle) -> None:
    perf = bundle.json("development/performance.json") or {}
    table = bundle.csv("development/scorecard_table.csv")
    p = perf.get("performance", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Gini validation (điểm nguyên)", num(p.get("validation", {}).get("points", {}).get("gini")))
    c2.metric("KS validation", num(p.get("validation", {}).get("points", {}).get("ks")))
    c3.metric("Gini train", num(p.get("train", {}).get("points", {}).get("gini")))

    bands = pd.DataFrame(perf.get("validation_bands") or [])
    if len(bands):
        bands["Score"] = bands["score_min"].astype(str) + " – " + bands["score_max"].astype(str)
        long = bands.melt(id_vars=["band", "Score"], value_vars=["bad_rate", "pd_mean"], var_name="series", value_name="rate")
        long["series"] = long["series"].map({"bad_rate": "Bad rate thực tế", "pd_mean": "PD từ thang điểm"})
        chart = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("band:O", title="Decile điểm (1 = điểm thấp nhất = rủi ro cao nhất)"),
            y=alt.Y("rate:Q", title="Tỷ lệ", axis=alt.Axis(format="%")),
            color=alt.Color("series:N", title=None), tooltip=["band", "Score", alt.Tooltip("rate:Q", format=".2%")],
        ).properties(height=280, title="Validation: bad rate theo decile điểm")
        st.altair_chart(chart, use_container_width=True)

    if table is not None:
        st.markdown("**Bảng điểm** (một dòng mỗi biến × bin; điểm cơ sở ở dòng đầu)")
        feats = [f for f in table["feature"].unique() if f != "(base points)"]
        pick = st.multiselect("Lọc biến", feats, default=feats[:5])
        shown = table[table["feature"].isin(pick)] if pick else table
        st.dataframe(shown.style.format({"woe": "{:.3f}", "coefficient": "{:.4f}", "train_share": "{:.1%}",
                                         "train_bad_rate": "{:.2%}"}, na_rep=""), hide_index=True, use_container_width=True)
    trail = bundle.csv("development/sign_trail.csv")
    if trail is not None and len(trail):
        with st.expander(f"Biến bị loại vì dấu hệ số sai hoặc không ổn định ({len(trail)})"):
            st.dataframe(trail, hide_index=True, use_container_width=True)
    card = bundle.text("development/model_card.md")
    if card:
        with st.expander("Model card đầy đủ"):
            st.markdown(card)


def _challenger(bundle: Bundle) -> None:
    perf = bundle.json("development/challenger_performance.json") or {}
    imp = bundle.csv("development/challenger_importance.csv")
    p = perf.get("performance", {})
    champ = perf.get("champion", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Gini validation, LightGBM", num((p.get("validation") or {}).get("gini")))
    c2.metric("Gini validation, scorecard", num((champ.get("validation") or {}).get("gini")))
    agree = perf.get("agreement") or {}
    c3.metric("Trùng top 10% rủi ro với champion", pct(agree.get("top_overlap")),
              help=f"Spearman giữa hai thứ hạng rủi ro: {num(agree.get('spearman'))}")
    if imp is not None:
        top = imp.head(20)
        chart = alt.Chart(top).mark_bar(color="#d9480f").encode(
            y=alt.Y("feature:N", sort=None, title=None), x=alt.X("mean_abs_shap:Q", title="Mean |SHAP| (log-odds)"),
            tooltip=["rank", "feature", alt.Tooltip("mean_abs_shap:Q", format=".3f"), alt.Tooltip("gain_share:Q", format=".1%")],
        ).properties(height=24 * len(top), title="20 biến quan trọng nhất (TreeSHAP của LightGBM)")
        st.altair_chart(chart, use_container_width=True)
    trials = bundle.csv("development/challenger_trials.csv")
    if trials is not None:
        with st.expander(f"Các cấu hình đã thử ({len(trials)})"):
            st.dataframe(trials, hide_index=True, use_container_width=True)
    report = bundle.text("development/challenger_report.md")
    if report:
        with st.expander("Báo cáo challenger đầy đủ"):
            st.markdown(report)


def _calibration(bundle: Bundle) -> None:
    cal = bundle.json("development/calibration.json")
    pkg = bundle.json("development/package.json") or {}
    if not cal:
        st.info("Không có calibration.json trong bundle.")
        return
    rows, rel = [], []
    for model in ("champion", "challenger"):
        c = cal.get(model) or {}
        after = (c.get("after") or {}).get("validation") or {}
        before = (c.get("before") or {}).get("validation") or {}
        rows.append({"Mô hình": model, "Phương pháp": c.get("chosen"), "Slope trước": before.get("slope"),
                     "Slope sau": after.get("slope"), "Intercept sau": after.get("intercept"),
                     "Brier sau": after.get("brier"), "PD TB sau": after.get("mean_pd"), "Bad rate": after.get("bad_rate")})
        for r in c.get("reliability") or []:
            rel.append({**r, "model": model})
    st.dataframe(pd.DataFrame(rows).style.format({"Slope trước": "{:.3f}", "Slope sau": "{:.3f}", "Intercept sau": "{:.3f}",
                                                  "Brier sau": "{:.4f}", "PD TB sau": "{:.2%}", "Bad rate": "{:.2%}"}, na_rep=""),
                 hide_index=True, use_container_width=True)
    if rel:
        df = pd.DataFrame(rel)
        base = alt.Chart(df).encode(
            x=alt.X("mean_pd:Q", title="PD trung bình của band", axis=alt.Axis(format="%")),
            y=alt.Y("bad_rate:Q", title="Bad rate quan sát", axis=alt.Axis(format="%")),
            color=alt.Color("model:N", scale=model_scale(), title=None),
            tooltip=["model", "band", "n", alt.Tooltip("mean_pd:Q", format=".2%"), alt.Tooltip("bad_rate:Q", format=".2%")])
        diag = alt.Chart(pd.DataFrame({"x": [0, df["mean_pd"].max()]})).mark_line(strokeDash=[4, 4], color="grey").encode(
            x="x:Q", y="x:Q")
        st.altair_chart((base.mark_line(point=True) + diag).properties(height=320,
                        title="Reliability trên validation sau calibration (đường chéo = hoàn hảo)"), use_container_width=True)
    st.caption(f"Gói model đóng băng `{(pkg.get('fingerprint') or '')[:12]}` lúc {pkg.get('frozen_at', '')}, "
               f"fit calibration trên mẫu `{pkg.get('fitted_on', '')}`.")
    report = bundle.text("development/calibration_report.md")
    if report:
        with st.expander("Báo cáo calibration đầy đủ"):
            st.markdown(report)
