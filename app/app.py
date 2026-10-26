"""Dashboard demo - Credit Risk Model Lifecycle Platform.

    streamlit run app/app.py

Reads the bundle written by the `report` pipeline step (reports/latest by default). Pages follow
the business decision flow of the three-minute demo (OVERVIEW 7.1): problem -> build -> validate
-> monitor -> early warning -> engineering -> limitations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Bundle, resolve_bundle  # noqa: E402
from views import development, engineering, ews, limitations, monitoring, overview, validation  # noqa: E402

st.set_page_config(page_title="Credit Risk Lifecycle", page_icon="📊", layout="wide")

folder = resolve_bundle()
if folder is None:
    st.title("Credit Risk Model Lifecycle Platform")
    st.error("Chưa có bundle báo cáo. Chạy pipeline rồi bước `report`:")
    st.code("python -m credit_risk run\npython -m credit_risk run --steps report", language="powershell")
    st.stop()

bundle = Bundle(folder)

PAGES = [  # (title, url slug, renderer) in demo order
    ("Tổng quan", "tong-quan", overview.render),
    ("Xây dựng mô hình", "xay-dung-mo-hinh", development.render),
    ("Thẩm định độc lập", "tham-dinh-doc-lap", validation.render),
    ("Giám sát danh mục", "giam-sat-danh-muc", monitoring.render),
    ("Cảnh báo sớm", "canh-bao-som", ews.render),
    ("Kỹ thuật", "ky-thuat", engineering.render),
    ("Giới hạn", "gioi-han", limitations.render),
]

pages = [st.Page(lambda fn=fn: fn(bundle), title=title, url_path=slug, default=(i == 0))
         for i, (title, slug, fn) in enumerate(PAGES)]
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.caption(f"Run `{bundle.index.get('run_id', '')}` · config `{bundle.index.get('config_hash', '')[:12]}`")
    st.caption(f"Tạo lúc {bundle.index.get('generated_at', '')}")

nav.run()
