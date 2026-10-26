"""Page 6 - engineering: the pipeline, lineage of this bundle, environment, how to run and test."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import REPO_ROOT, Bundle

STEPS = [
    ("ingest", "CSV / TXT thô → parquet đúng kiểu", "data/processed/<source>/*.parquet"),
    ("validate-data", "Kiểm tra theo data contract; rule error fail thì dừng", "artifacts/<run>/data_validation/"),
    ("features", "Split, availability matrix, feature, binning + WoE fit trên train, IV, shortlist", "features.parquet, binning.json, shortlist.json"),
    ("train", "Scorecard (logistic trên WoE, kiểm dấu, quy đổi điểm, mã lý do, model card) + LightGBM challenger (SHAP)", "artifacts/<run>/train/"),
    ("calibrate", "Platt / isotonic chọn bằng CV; đóng băng model package", "data/processed/home_credit/model_package/"),
    ("validate-model", "Thẩm định độc lập: tính lại mọi chỉ số, PSI, bootstrap, phân khúc, so sánh, đối chiếu triển khai; kết luận", "metrics.json, validation_report.md"),
    ("monitor", "Panel, chỉ số danh mục, vintage, roll rate / migration, tập trung, cảnh báo", "monitoring_metrics.json, *.csv"),
    ("ews", "Nhãn 90+ trong 3 tháng, feature tại t, baseline + LightGBM, watchlist, survival", "ews_metrics.json, watchlist.csv"),
    ("report", "Gom output mới nhất của mọi bước thành bundle cho dashboard này", "artifacts/<run>/report/, reports/latest/"),
]


def render(bundle: Bundle) -> None:
    st.title("Kỹ thuật")
    st.markdown("Toàn bộ project chạy bằng **một lệnh**, không phụ thuộc notebook. Config YAML là single source of truth; "
                "mỗi lần chạy sinh một `run_id` ghi lại config hash, seed, git commit và phiên bản thư viện.")
    st.code("python -m credit_risk run                       # đủ 9 bước, cả hai nguồn\n"
            "python -m credit_risk run --steps report        # chỉ gom bundle cho dashboard\n"
            "python -m pytest                                # bộ test\n"
            "streamlit run app/app.py                        # dashboard này", language="powershell")

    st.subheader("Chín bước của pipeline")
    st.dataframe(pd.DataFrame(STEPS, columns=["Bước", "Làm gì", "Để lại gì"]), hide_index=True, use_container_width=True)

    st.subheader("Lineage của bundle này")
    manifests = bundle.index.get("run_manifests") or {}
    rows = []
    for step, rid in bundle.runs.items():
        mf = manifests.get(step) or {}
        git = mf.get("git") or {}
        rows.append({"Bước": step, "Run": rid or "—", "Config hash": (mf.get("config_hash") or "")[:12],
                     "Git commit": (git.get("commit") or "")[:10] + (" (dirty)" if git.get("dirty") else ""),
                     "Bắt đầu": mf.get("started_at") or "", "Python": mf.get("python") or ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    pk = {}
    for mf in manifests.values():
        pk = mf.get("packages") or pk
    if pk:
        st.caption("Thư viện: " + ", ".join(f"{k} {v}" for k, v in pk.items() if v))
    missing = bundle.index.get("missing") or {}
    if missing:
        with st.expander("File thiếu trong bundle"):
            st.json(missing)

    st.subheader("Năm nguyên tắc vận hành")
    st.markdown(
        "1. **Config là single source of truth**: ngưỡng DPD, cửa sổ ba tháng, tham số quy đổi điểm, ngưỡng cảnh báo đều nằm trong YAML.\n"
        "2. **Mọi kết quả có lineage**: nhìn một con số là lần ra được run, config hash, commit sinh ra nó.\n"
        "3. **Mọi transformation chỉ fit trên train**, rồi áp sang chỗ khác.\n"
        "4. **Time-based split là mặc định** với dữ liệu panel; Home Credit là ngoại lệ có ghi nhận (không có ngày nộp đơn).\n"
        "5. **Không commit dữ liệu thô**; `data/README.md` chỉ hướng dẫn tải.")

    st.subheader("Kiểm thử")
    tests_dir = REPO_ROOT / "tests"
    if tests_dir.exists():
        files = sorted(p.name for p in tests_dir.glob("test_*.py"))
        st.markdown(f"{len(files)} file test trong `tests/`; mỗi chặng có một file Definition of Done chạy pipeline thật trên dữ liệu tổng hợp "
                    "(`test_stage*_dod.py`), bên cạnh test đơn vị cho metric, nhãn, binning, data validation.")
        with st.expander("Danh sách file test"):
            st.code("\n".join(files))
    st.subheader("Cấu trúc")
    st.code(
        "src/credit_risk/\n"
        "├── data/        contracts, ingest, validate (DuckDB)\n"
        "├── features/    split, availability, build, binning, iv_report, selection, woe\n"
        "├── models/      scorecard (champion), challenger (LightGBM, SHAP), calibration\n"
        "├── validation/  package, population, discrimination, calibration_check, stability, uncertainty,\n"
        "│                segments, comparison, implementation, conclusion, engine\n"
        "├── monitoring/  buckets, transitions, panel, portfolio, vintage, roll, concentration, alerts, engine\n"
        "├── ews/         labels, cohort, features, models, metrics, evaluation, watchlist, survival, engine\n"
        "├── reporting/   model_card, challenger_report, calibration_report, bundle\n"
        "├── pipeline.py  9 bước, run_id, manifest\n"
        "└── cli.py\n"
        "app/             dashboard này (chỉ đọc bundle)\n"
        "configs/         definitions, model_dev, validation, ews, feature_availability, data_contracts/")
