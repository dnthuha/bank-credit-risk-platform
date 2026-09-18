# Project scope

## Dùng cho ai

- **Người đọc portfolio / nhà tuyển dụng** trong mảng credit risk, model validation, risk analytics: xem được trọn vòng đời của một mô hình trong chưa tới mười phút.
- **Chính người làm project**: một khung làm việc có kỷ luật, trong đó định nghĩa, dữ liệu, mô hình và kết luận đều truy ngược được.

## Cho ra cái gì

| Module | Câu hỏi nghiệp vụ | Deliverable |
|---|---|---|
| 1. Model Development (Home Credit) | Hồ sơ này rủi ro tới mức nào, từ chối thì vì sao? | Scorecard (champion), LightGBM (challenger), PD đã calibrate, reason codes, model card |
| 2. Independent Validation (Home Credit) | Model có đủ điều kiện phê duyệt không? | `metrics.json` và validation report tự động có kết luận |
| 3. Monitoring & EWS (Freddie Mac) | Danh mục xấu đi ở đâu, khoản nào sắp 90+ DPD? | Chỉ số danh mục hằng tháng, vintage / roll rate / migration, watchlist 3 tháng |

Cùng với đó: pipeline chạy bằng một lệnh, test tự động cho phần tính chỉ số và gán nhãn, cùng một dashboard demo.

## Cố tình không làm

- LGD, EAD, vốn kinh tế, IFRS 9 / CECL đầy đủ.
- Hệ thống phê duyệt tín dụng chạy thật, API production.
- Dữ liệu khách hàng thật hoặc dữ liệu nhạy cảm.
- Spark, Airflow, Kubernetes, Docker, MLflow trước khi phần lõi ổn định.
- Đua thứ hạng Kaggle bằng ensemble phức tạp.

## Quyết định đã chốt (2026-09-17)

| # | Vấn đề | Quyết định | Hệ quả phải ghi nhận |
|---|---|---|---|
| 1 | Home Credit không có ngày nộp đơn, `application_test` không có nhãn | Stratified random split train / validation / calibration / test (60/10/15/15). `application_test` là mẫu hiện tại để tính PSI. | **Không có out-of-time thật.** Ghi rõ trong model card và validation report. `application_test` có population khác train (Revolving loans 0.9% so với 9.5%), nên PSI phải tách theo `NAME_CONTRACT_TYPE`; xem [docs/data_quality_home_credit.md](docs/data_quality_home_credit.md). |
| 2 | Chọn Fannie Mae hay Freddie Mac | Freddie Mac **sample dataset**, vintage 2012-2019 | 50k khoản vay / năm là mẫu ngẫu nhiên, không phải toàn bộ danh mục. |
| 3 | Forbearance COVID 2020-2021 làm tỷ lệ quá hạn tăng vọt | 2020-03 → 2021-12 là **stress segment**: không train, đánh giá riêng | Out-of-time sau COVID bắt đầu từ 2022-01. |
| 4 | Laptop khoảng 7.4 GB RAM | DuckDB + parquet, giới hạn RAM của DuckDB qua `.env` | Bảng panel lớn phải xử lý trong DuckDB, không load hết vào pandas. |
| 5 | Cure rồi lại 90+ | Vẫn tính là event (`redefault_after_cure_counts_as_event: true`) | Watchlist bắt được cả khoản tái vỡ nợ. Có cờ `ever_90plus_before_t` để phân tích riêng. |
| 6 | Môi trường | `venv` + Python 3.13, thư viện ghim trong `requirements.txt` | Conda không solve được env trên máy này, nên không dùng. |

Mọi định nghĩa có thể đổi được nằm ở `configs/definitions.yaml`. Đổi định nghĩa sẽ đổi config hash ghi trong manifest của mỗi run.

## Entry criteria cho model development

Chỉ bắt đầu tinh chỉnh mô hình khi trả lời trôi chảy được ba câu:

1. **Một dòng là gì?** Home Credit: một hồ sơ tại ngày nộp đơn. Freddie Mac: một cặp khoản vay - tháng báo cáo.
2. **Target là gì, đo trong cửa sổ nào?** Home Credit: `TARGET` do nguồn định nghĩa. EWS: lần đầu vào 90+ DPD hoặc liquidation trong t+1..t+3.
3. **Tại thời điểm dự báo, thông tin nào đã có?** Home Credit: chỉ những gì có trước ngày nộp đơn (cần feature availability matrix ở chặng 2). EWS: chỉ các tháng ≤ t.
