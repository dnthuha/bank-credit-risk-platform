"""Page 3 - independent validation: verdicts first, then the recomputed numbers, then the findings."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from common import (LIGHT_COLOURS, VERDICT_COLOURS, Bundle, badge, light_scale, model_scale, num, pct,
                    section_missing, verdict_label)


def render(bundle: Bundle) -> None:
    st.title("Thẩm định độc lập")
    st.markdown("Code thẩm định đọc **gói model đã đóng băng** rồi tính lại mọi chỉ số từ đầu: AUC / Gini / KS với "
                "khoảng tin cậy bootstrap, hiệu chỉnh, PSI, phân khúc (kể cả nhóm nhạy cảm), so sánh hai mô hình và "
                "đối chiếu triển khai. Kết luận phê duyệt sinh tự động từ quy tắc trong `validation.yaml`.")
    if section_missing("validation", bundle, "validation/metrics.json"):
        return
    m = bundle.json("validation/metrics.json") or {}
    p, sections, conclusion = m.get("provenance", {}), m.get("sections", {}), m.get("conclusion") or {}
    sample = p.get("headline_sample", "test")

    # --- verdicts ---------------------------------------------------------
    c1, c2, c3 = st.columns([1, 1, 2])
    for col, model in ((c1, "champion"), (c2, "challenger")):
        v = (conclusion.get("verdicts") or {}).get(model) or {}
        code = v.get("code") if isinstance(v, dict) else v
        col.markdown(f"### {model}")
        col.markdown(badge(verdict_label(code), VERDICT_COLOURS.get(code, "#607d8b")), unsafe_allow_html=True)
    rec = conclusion.get("recommendation") or {}
    with c3:
        st.markdown("### Khuyến nghị")
        st.markdown(rec.get("text") or "Chưa có quy tắc kết luận (chặng 7).")
    st.caption(f"{p.get('checks_total')} kiểm tra tính lại, {p.get('checks_failed')} lệch, {p.get('findings')} phát hiện. "
               f"Mẫu chính: {sample}. Bootstrap {p.get('bootstrap_resamples')} lần, seed {p.get('seed')}.")

    rules = pd.DataFrame(conclusion.get("rules") or [])
    if len(rules):
        with st.expander("Mọi quy tắc kết luận và kết quả"):
            st.dataframe(rules, hide_index=True, use_container_width=True)

    tabs = st.tabs(["Sức phân biệt", "Hiệu chỉnh", "Ổn định (PSI)", "Phân khúc và fairness", "Hai mô hình", "Phát hiện", "Báo cáo"])
    with tabs[0]:
        _discrimination(sections, sample)
    with tabs[1]:
        _calibration(sections, sample)
    with tabs[2]:
        _stability(sections)
    with tabs[3]:
        _segments(sections, sample)
    with tabs[4]:
        _comparison(sections, sample)
    with tabs[5]:
        _findings(m)
    with tabs[6]:
        report = bundle.text("validation/validation_report.md")
        if report:
            st.markdown(report)


def _discrimination(sections: dict, sample: str) -> None:
    disc = pd.DataFrame((sections.get("discrimination") or {}).get("table") or [])
    unc = pd.DataFrame((sections.get("uncertainty") or {}).get("intervals") or [])
    if len(unc):
        g = unc[unc["metric"].isin(["auc", "gini", "ks"])].copy()
        points = alt.Chart().mark_point(filled=True, size=90).encode(
            x=alt.X("estimate:Q", title="Ước lượng với khoảng tin cậy 95% (bootstrap)", scale=alt.Scale(zero=False)),
            y=alt.Y("split:N", title=None), color=alt.Color("model:N", scale=model_scale(), title=None),
            yOffset="model:N", tooltip=["model", "split", "metric", alt.Tooltip("estimate:Q", format=".4f"),
                                        alt.Tooltip("ci_low:Q", format=".4f"), alt.Tooltip("ci_high:Q", format=".4f")],
        )
        bars = alt.Chart().mark_rule().encode(x="ci_low:Q", x2="ci_high:Q", y="split:N", yOffset="model:N",
                                              color=alt.Color("model:N", scale=model_scale()))
        st.altair_chart(alt.layer(bars, points, data=g).properties(width=270, height=160)
                        .facet(column=alt.Column("metric:N", title=None)).resolve_scale(x="independent"),
                        use_container_width=False)
    if len(disc):
        show = disc[["model", "split", "n", "bads", "bad_rate", "auc", "gini", "ks", "ks_cut"]]
        st.dataframe(show.style.format({"bad_rate": "{:.2%}", "auc": "{:.4f}", "gini": "{:.4f}", "ks": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
    diff = pd.DataFrame((sections.get("uncertainty") or {}).get("differences") or [])
    if len(diff):
        st.markdown("**Chênh lệch có ý nghĩa không?** (bootstrap ghép cặp, cùng mẫu)")
        st.dataframe(diff[["comparison", "split", "metric", "estimate", "ci_low", "ci_high", "excludes_zero"]]
                     .style.format({"estimate": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}"}),
                     hide_index=True, use_container_width=True)


def _calibration(sections: dict, sample: str) -> None:
    summary = pd.DataFrame((sections.get("calibration") or {}).get("summary") or [])
    bands = pd.DataFrame((sections.get("calibration") or {}).get("bands") or [])
    if len(summary):
        head = summary[summary["split"] == sample]
        cols = ["model", "split", "n", "mean_pd", "bad_rate", "ratio_bad_rate_to_pd", "slope", "slope_se", "intercept_in_the_large",
                "brier", "hl_p_value", "bands_flagged", "bands_flagged_list"]
        st.dataframe(head[[c for c in cols if c in head]].style.format(
            {"mean_pd": "{:.3%}", "bad_rate": "{:.3%}", "ratio_bad_rate_to_pd": "{:.3f}", "slope": "{:.3f}", "slope_se": "{:.3f}",
             "intercept_in_the_large": "{:.3f}", "brier": "{:.4f}", "hl_p_value": "{:.3f}"}, na_rep=""),
            hide_index=True, use_container_width=True)
    if len(bands):
        b = bands[bands["split"] == sample]
        base = alt.Chart(b).encode(x=alt.X("band:O", title="Band PD (1 = thấp nhất)"))
        exp = base.mark_bar(opacity=0.35).encode(y=alt.Y("expected:Q", title="Số bad"), color=alt.Color("model:N", scale=model_scale()),
                                                 xOffset="model:N", tooltip=["model", "band", "n", "observed", alt.Tooltip("expected:Q", format=".1f"), alt.Tooltip("z:Q", format=".2f"), "flag"])
        obs = base.mark_point(filled=True, size=70, color="black").encode(y="observed:Q", xOffset="model:N")
        st.altair_chart((exp + obs).properties(height=300, title=f"{sample}: số bad kỳ vọng theo PD (cột) so với quan sát (chấm)"),
                        use_container_width=True)
        flagged = b[b["flag"].astype(str) != ""]
        if len(flagged):
            st.caption("Band lệch hệ thống: " + "; ".join(f"{r.model} band {r.band} ({r.flag}, z {r.z:+.1f})" for r in flagged.itertuples()))


def _stability(sections: dict) -> None:
    table = pd.DataFrame((sections.get("stability") or {}).get("table") or [])
    if not len(table):
        st.info("Không có bảng PSI.")
        return
    out = table[table["kind"] == "output"]
    feat = table[table["kind"] != "output"]
    st.markdown("**PSI của output** (score / PD) so với mẫu tham chiếu train; application_test là dân số mới, không có nhãn.")
    st.dataframe(out[["model", "name", "current", "segment", "n_reference", "n_current", "psi", "light"]]
                 .style.format({"psi": "{:.4f}"}), hide_index=True, use_container_width=True)
    if len(feat):
        worst = feat.sort_values("psi", ascending=False).head(20)
        chart = alt.Chart(worst).mark_bar().encode(
            y=alt.Y("name:N", sort=None, title=None), x=alt.X("psi:Q", title="PSI"),
            color=alt.Color("light:N", scale=light_scale(), title=None),
            tooltip=["model", "name", "current", "segment", alt.Tooltip("psi:Q", format=".4f"), "missing_reference", "missing_current"],
        ).properties(height=24 * len(worst), title="Biến đầu vào có PSI cao nhất (< 0.10 ổn định, 0.10–0.25 để mắt, > 0.25 đáng kể)")
        st.altair_chart(chart, use_container_width=True)


def _segments(sections: dict, sample: str) -> None:
    table = pd.DataFrame((sections.get("segments") or {}).get("table") or [])
    if not len(table):
        st.info("Không có bảng phân khúc.")
        return
    fair = table[(table["fairness_review"] == True) & (table["group"] != "(all)")]  # noqa: E712
    st.markdown("**Nhóm nhạy cảm** (PROJECT_SCOPE #7: giới tính, tình trạng hôn nhân, tuổi không bị loại theo chính sách, "
                "nên hiệu năng và tỷ lệ từ chối phải được soi theo từng nhóm). AIR = tỷ lệ duyệt của nhóm / nhóm cao nhất; "
                "dưới 0.80 là cờ theo quy tắc 4/5.")
    if len(fair):
        chart = alt.Chart().mark_bar().encode(
            x=alt.X("adverse_impact_ratio:Q", title="Adverse impact ratio"), y=alt.Y("group:N", title=None),
            color=alt.Color("model:N", scale=model_scale(), title=None), yOffset="model:N",
            tooltip=["segment", "group", "model", "n", alt.Tooltip("reject_rate:Q", format=".1%"),
                     alt.Tooltip("adverse_impact_ratio:Q", format=".2f"), alt.Tooltip("auc:Q", format=".3f"), "calibration_flag"],
        ).properties(height=22 * fair["group"].nunique() * 2)
        rule = alt.Chart().mark_rule(color="#c62828", strokeDash=[4, 4]).encode(x=alt.datum(0.8))
        st.altair_chart(alt.layer(chart, rule, data=fair).facet(row=alt.Row("segment:N", title=None))
                        .resolve_scale(y="independent"), use_container_width=True)
    cols = ["segment", "group", "model", "n", "bad_rate", "mean_pd", "calibration_z", "calibration_flag", "reject_rate",
            "adverse_impact_ratio", "auc", "auc_gap_vs_all"]
    st.dataframe(table[[c for c in cols if c in table]].style.format(
        {"bad_rate": "{:.2%}", "mean_pd": "{:.2%}", "calibration_z": "{:+.2f}", "reject_rate": "{:.1%}",
         "adverse_impact_ratio": "{:.2f}", "auc": "{:.3f}", "auc_gap_vs_all": "{:+.3f}"}, na_rep=""),
        hide_index=True, use_container_width=True, height=400)


def _comparison(sections: dict, sample: str) -> None:
    cmp = sections.get("comparison") or {}
    table = pd.DataFrame(cmp.get("table") or [])
    if len(table):
        st.dataframe(table.set_index("model").T, use_container_width=True)
    dec = cmp.get("decisions") or {}
    eq = cmp.get("equal_volume") or {}
    if dec and eq:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Quyết định ở ngưỡng từ chối hiện tại ({sample})**")
            st.dataframe(pd.DataFrame(dec).T.style.format({"reject_rate": "{:.1%}", "bad_rate_approved": "{:.2%}", "bads_caught_share": "{:.1%}"}),
                         use_container_width=True)
        with c2:
            st.markdown("**Cùng khối lượng từ chối**")
            st.dataframe(pd.DataFrame(eq).T.style.format({"reject_rate": "{:.1%}", "bad_rate_approved": "{:.2%}", "bads_caught_share": "{:.1%}"}),
                         use_container_width=True)
    swap = cmp.get("swap_set")
    if isinstance(swap, list) and swap:
        st.markdown("**Swap set**: hồ sơ hai mô hình quyết định khác nhau ở ngưỡng từ chối")
        st.dataframe(pd.DataFrame(swap).style.format({"share": "{:.1%}", "bad_rate": "{:.2%}"}), hide_index=True, use_container_width=True)
    agree = cmp.get("agreement") or {}
    if agree.get(sample):
        a = agree[sample]
        st.caption(f"Spearman giữa hai thứ hạng {a.get('spearman'):.3f}; trùng top {a.get('top_share'):.0%} rủi ro "
                   f"{a.get('top_overlap'):.1%}; bad rate nhóm chỉ champion xếp top {pct(a.get('bad_rate_only_champion_top'))}, "
                   f"chỉ challenger xếp top {pct(a.get('bad_rate_only_challenger_top'))}.")


def _findings(m: dict) -> None:
    f = pd.DataFrame(m.get("findings") or [])
    if not len(f):
        st.success("Không có phát hiện.")
        return
    order = {"blocking": 0, "major": 1, "fairness": 2, "note": 3}
    f["_o"] = f["severity"].map(order).fillna(9)
    f = f.sort_values(["_o", "module"]).drop(columns="_o")
    colours = {"blocking": "#c62828", "major": "#e65100", "fairness": "#6a1b9a", "note": "#607d8b"}
    for r in f.itertuples(index=False):
        st.markdown(f"{badge(r.severity, colours.get(r.severity, '#607d8b'))} **{r.module}** · {r.finding}", unsafe_allow_html=True)
    failed = pd.DataFrame(m.get("checks") or [])
    if len(failed) and "ok" in failed:
        failed = failed[failed["ok"] == False]  # noqa: E712
        if len(failed):
            st.markdown("**Kiểm tra lệch**")
            st.dataframe(failed, hide_index=True, use_container_width=True)
