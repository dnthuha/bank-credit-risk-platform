"""Page 1 - the problem and the four answers, on one screen."""

from __future__ import annotations

import streamlit as st

from common import LIGHT_COLOURS, VERDICT_COLOURS, Bundle, badge, ci, light_label, num, pct, short, verdict_label


def render(bundle: Bundle) -> None:
    h = bundle.headline
    st.title("Credit Risk Model Lifecycle Platform")
    st.markdown(
        "Một ngân hàng cần **cả vòng đời** của mô hình rủi ro tín dụng: xây mô hình, thẩm định độc lập, "
        "giám sát danh mục và phát hiện sớm khoản sắp thành nợ xấu. Nền tảng này làm trọn bốn việc đó trên hai "
        "bộ dữ liệu công khai, bằng một pipeline chạy từ config, mọi con số đều truy ngược được về `run_id`."
    )
    st.markdown(
        "```\nModel Development  ->  Independent Validation  ->  Portfolio Monitoring  ->  Early Warning\n"
        "   (Home Credit)            (Home Credit)                (Freddie Mac)            (Freddie Mac)\n```"
    )

    c1, c2, c3, c4 = st.columns(4)
    dev, val, mon, ews = h.get("development"), h.get("validation"), h.get("monitoring"), h.get("ews")

    with c1:
        st.subheader("1 · Xây dựng")
        if dev:
            st.metric("Gini validation, scorecard", num(dev.get("gini_validation_champion")))
            st.metric("Gini validation, LightGBM", num(dev.get("gini_validation_challenger")))
            st.caption(f"Scorecard {dev.get('champion_features')} biến, điểm cơ sở {dev.get('base_points')}; "
                       "mỗi hồ sơ có mã lý do.")
        else:
            st.info("Chưa có kết quả train.")

    with c2:
        st.subheader("2 · Thẩm định")
        if val:
            models = val.get("models", {})
            for m in ("champion", "challenger"):
                v = (models.get(m) or {}).get("verdict") or {}
                code = v.get("code") if isinstance(v, dict) else v
                st.markdown(f"**{m}** {badge(verdict_label(code), VERDICT_COLOURS.get(code, '#607d8b'))}",
                            unsafe_allow_html=True)
                st.caption(f"Gini {val.get('sample')} {num((models.get(m) or {}).get('gini'))} "
                           f"{ci((models.get(m) or {}).get('gini_ci'))}")
            st.caption(f"{val.get('checks_total')} kiểm tra tính lại, {val.get('checks_failed')} lệch, "
                       f"{val.get('findings')} phát hiện.")
        else:
            st.info("Chưa có kết quả validate-model.")

    with c3:
        st.subheader("3 · Giám sát")
        if mon:
            status = mon.get("status")
            st.markdown(f"Tháng **{mon.get('reporting_month')}** "
                        f"{badge(light_label(status), LIGHT_COLOURS.get(status, '#607d8b'))}", unsafe_allow_html=True)
            st.metric("90+ DPD (theo khoản)", pct(mon.get("dpd90plus_rate_count"), 3))
            alerts = mon.get("alerts") or {}
            st.caption(f"{alerts.get('red', 0)} cảnh báo đỏ, {alerts.get('amber', 0)} vàng trên "
                       f"{(mon.get('panel') or {}).get('loans', 0):,} khoản.")
        else:
            st.info("Chưa có kết quả monitor.")

    with c4:
        st.subheader("4 · Cảnh báo sớm")
        if ews:
            oot = ews.get("out_of_time") or {}
            w = ews.get("watchlist") or {}
            cap_key = next((k for k in oot if k.startswith("capture_top")), None)
            lift_key = next((k for k in oot if k.startswith("lift_top")), None)
            st.metric(f"Capture out-of-time ({cap_key.replace('capture_top_', 'top ').replace('pct', '%') if cap_key else 'top'})",
                      pct(oot.get(cap_key)) if cap_key else "—")
            if lift_key:
                st.metric("Lift so với ngẫu nhiên", f"{oot.get(lift_key):.1f}×" if oot.get(lift_key) else "—")
            st.caption(f"Watchlist {ews.get('scoring_month')}: {w.get('listed', 0):,} khoản "
                       f"({pct(w.get('listed_share'), 0)} cohort), kỳ vọng {w.get('expected_events', 0):.0f} khoản vào 90+ "
                       f"trong {ews.get('horizon_months')} tháng.")
        else:
            st.info("Chưa có kết quả ews.")

    st.divider()
    if val and val.get("recommendation"):
        rec = val["recommendation"]
        st.subheader("Kết luận phê duyệt (tự động, theo quy tắc trong `validation.yaml`)")
        st.markdown(rec.get("text", ""))
        gap = rec.get("gini_gap_test") or {}
        if gap:
            st.caption(f"Chênh Gini challenger − champion trên {val.get('sample')}: {num(gap.get('estimate'))} "
                       f"{ci([gap.get('ci_low'), gap.get('ci_high')])} (bootstrap ghép cặp"
                       f"{', khác 0' if gap.get('excludes_zero') else ''}).")

    st.subheader("Lineage")
    rows = [{"Bước": step, "Run": rid or "—",
             "Git": short((((bundle.index.get("run_manifests") or {}).get(step) or {}).get("git") or {}).get("commit"), 10)
             if rid else ""}
            for step, rid in bundle.runs.items()]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(f"Gói model đã đóng băng `{short((val or {}).get('package_fingerprint'))}`. "
               "Dashboard chỉ đọc artifact, không tính lại chỉ số.")
