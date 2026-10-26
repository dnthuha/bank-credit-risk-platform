"""Page 7 - limitations, gathered from every module's own report, and what comes next."""

from __future__ import annotations

import streamlit as st

from common import Bundle

NEXT_STEPS = [
    "**Out-of-time thật cho Module 1**: Home Credit không có ngày nộp đơn, nên hiệu năng test là cùng kỳ. Một nguồn có ngày "
    "nộp đơn (hoặc một proxy thời gian) sẽ cho phép đo trôi hiệu năng theo thời gian như đã làm với Freddie Mac.",
    "**Fairness**: champion Phải khắc phục vì AIR 0.56 ở nhóm dưới 30 tuổi. Hướng xử lý: bỏ hoặc thay các biến nhạy cảm, hoặc "
    "hiệu chỉnh theo nhóm, rồi thẩm định lại; đây là lý do kết luận khuyến nghị challenger kèm điều kiện.",
    "**EWS**: thêm biến vĩ mô (lãi suất, giá nhà, thất nghiệp) và hiệu chỉnh lại PD theo đoạn thời gian; mô hình hiện học rất ít "
    "từ forbearance / modification vì chúng hiếm trong train.",
    "**Survival**: hiện chỉ mô tả (Kaplan–Meier, Aalen–Johansen); bước tiếp là mô hình có covariate (Cox) và rủi ro cạnh tranh "
    "đầy đủ giữa default và trả trước.",
    "**Ngoài phạm vi, có chủ đích**: LGD / EAD / IFRS 9, hệ thống phê duyệt chạy thật, API production, Spark / Airflow / MLflow.",
]


def render(bundle: Bundle) -> None:
    st.title("Giới hạn và hướng đi tiếp")
    st.markdown("Mỗi module tự ghi giới hạn của mình vào báo cáo; trang này gom lại để người đọc thấy ngay điều gì "
                "**không** nên suy ra từ các con số ở trên.")

    sources = [
        ("Thẩm định (Home Credit)", (bundle.json("validation/metrics.json") or {}).get("limitations")),
        ("Giám sát danh mục (Freddie Mac)", (bundle.json("monitoring/monitoring_metrics.json") or {}).get("limitations")),
        ("Cảnh báo sớm (Freddie Mac)", (bundle.json("ews/ews_metrics.json") or {}).get("limitations")),
    ]
    for title, items in sources:
        st.subheader(title)
        if items:
            for x in items:
                st.markdown(f"- {x}")
        else:
            st.caption("Không có trong bundle.")

    st.subheader("Hướng đi tiếp")
    for x in NEXT_STEPS:
        st.markdown(f"- {x}")
