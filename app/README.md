# app/

Dashboard demo (Streamlit), chặng 10. App **chỉ đọc** bundle do bước `report` của pipeline sinh ra
(`reports/latest/` mặc định, hoặc bất kỳ `artifacts/<run_id>/report/` chọn ở sidebar), không tự tính lại chỉ số
và không đọc dữ liệu.

```powershell
python -m credit_risk run --steps report     # gom output mới nhất của mọi bước
streamlit run app/app.py
```

| File | Vai trò |
|---|---|
| `app.py` | Điểm vào: tìm bundle, dựng navigation 7 trang theo luồng demo (OVERVIEW 7.1) |
| `common.py` | Tìm bundle (`CREDIT_RISK_BUNDLE_DIR`, `?bundle=`, sidebar), đọc file có cache, định dạng số |
| `views/overview.py` | Vấn đề và bốn câu trả lời, kết luận phê duyệt, lineage |
| `views/development.py` | Một hồ sơ cụ thể (điểm từng biến, mã lý do, PD hai mô hình), scorecard, challenger, calibration |
| `views/validation.py` | Kết luận từng mô hình, khuyến nghị, chỉ số với khoảng tin cậy, hiệu chỉnh, PSI, fairness, so sánh, phát hiện |
| `views/monitoring.py` | Trạng thái và cảnh báo, chuỗi tháng theo giai đoạn COVID, vintage, roll rate, migration, tập trung |
| `views/ews.py` | Watchlist (lọc, tải), backtest theo đoạn và theo tháng, phân khúc, calibration, survival, mô hình |
| `views/engineering.py` | Chín bước pipeline, lineage của bundle, thư viện, kiểm thử, cấu trúc |
| `views/limitations.py` | Giới hạn gom từ báo cáo của từng module, hướng đi tiếp |

Test: `tests/test_stage10_dod.py` render từng trang bằng `streamlit.testing.v1.AppTest` trên bundle sinh từ dữ liệu
tổng hợp, và kiểm tra app nói rõ khi thiếu bundle hoặc thiếu module.
