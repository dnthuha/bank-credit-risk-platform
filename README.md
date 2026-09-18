# bank-credit-risk-platform

Nền tảng bao quát toàn bộ **credit risk model lifecycle**: từ một PD scorecard và một machine learning challenger, đến independent validation, portfolio monitoring và Early Warning System cho các khoản vay có nguy cơ chuyển sang 90+ DPD.

```
Model Development  ->  Independent Model Validation  ->  Portfolio Monitoring  ->  Early Warning
   (Home Credit)            (Home Credit)                 (Freddie Mac)            (Freddie Mac)
```

- Phạm vi, quyết định đã chốt và các giới hạn: [PROJECT_SCOPE.md](PROJECT_SCOPE.md)
- Cách tải dữ liệu: [data/README.md](data/README.md)
- Chất lượng dữ liệu: [Home Credit](docs/data_quality_home_credit.md), [Freddie Mac](docs/data_quality_freddie_mac.md)
- Định nghĩa nghiệp vụ (single source of truth): [configs/definitions.yaml](configs/definitions.yaml)

## Trạng thái

| Chặng | Nội dung | Trạng thái |
|---|---|---|
| 1 | Nền móng: repo, môi trường, data contract, định nghĩa, ingest, data validation, test lõi | Dữ liệu thật đã qua validate: Home Credit (135 check, 0 error), Freddie Mac 2012-2025 (0 error) |
| 2 | WoE / IV, binning, rà soát leakage | |
| 3 | Scorecard | |
| 4 | LightGBM challenger | |
| 5 | Calibration, đóng băng model | |
| 6 | Validation engine | |
| 7 | Validation report tự động | |
| 8 | Portfolio monitoring | |
| 9 | Survival và EWS | |
| 10 | Dashboard, demo | |

## Cài đặt (Windows, Python 3.13)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
copy .env.example .env
```

Kiểm tra:

```powershell
python -m pytest
python -m credit_risk check-config
```

## Chạy pipeline

```powershell
python -m credit_risk run                                  # đủ 9 bước, cả hai nguồn
python -m credit_risk run --steps ingest validate-data     # chỉ một số bước
python -m credit_risk run --source freddie_mac             # chỉ một nguồn
```

| # | Bước | Làm gì | Output |
|---|---|---|---|
| 1 | `ingest` | CSV / TXT thô → parquet đúng kiểu | `data/processed/<source>/*.parquet` |
| 2 | `validate-data` | Kiểm tra theo data contract; rule `error` fail thì dừng | `artifacts/<run_id>/data_validation/` |
| 3 | `features` | Binning + WoE (M1), panel + biến trễ (M3) | *stub, chặng 2 và 8-9* |
| 4 | `train` | Scorecard + LightGBM | *stub, chặng 3-4* |
| 5 | `calibrate` | Platt / isotonic trên calibration sample | *stub, chặng 5* |
| 6 | `validate-model` | Tính lại độc lập AUC/Gini/KS, PSI, bootstrap | *stub, chặng 6* |
| 7 | `monitor` | Vintage, roll rate, migration | *stub, chặng 8* |
| 8 | `ews` | Cohort, nhãn 3 tháng, model, watchlist | *stub, chặng 9* |
| 9 | `report` | Model card, validation report, dữ liệu dashboard | *stub, chặng 7 và 10* |

Mỗi lần chạy sinh một `run_id` (`<UTC time>-<config hash>`). File `artifacts/<run_id>/run_manifest.json` ghi lại config hash, seed, git commit, phiên bản thư viện và trạng thái từng bước.

## Cấu trúc

```
bank-credit-risk-platform/
├── configs/
│   ├── definitions.yaml          # DPD, default, cửa sổ, COVID, scaling, ngưỡng
│   ├── model_dev.yaml            # split, binning, champion / challenger
│   ├── validation.yaml           # bootstrap, PSI, segment, implementation check
│   ├── ews.yaml                  # vintage, time split, alert capacity
│   └── data_contracts/           # data dictionary + rule kiểm tra cho từng nguồn
├── data/README.md                # chỉ hướng dẫn tải
├── docs/                         # data quality, phát hiện từ dữ liệu thật
├── src/credit_risk/
│   ├── data/                     # contracts, ingest, validate (DuckDB)
│   ├── features/                 # woe.py
│   ├── models/
│   ├── validation/               # metrics.py: AUC, Gini, KS, Brier, PSI
│   ├── monitoring/               # buckets.py, transitions.py (roll rate)
│   ├── ews/                      # labels.py: nhãn 90+ DPD trong 3 tháng
│   ├── reporting/
│   ├── pipeline.py               # 9 bước, run_id, manifest
│   └── cli.py
├── tests/                        # metric, label, data validation, pipeline, config
├── app/  notebooks/  reports/
└── artifacts/                    # sinh ra khi chạy, đã gitignore
```

## Nguyên tắc vận hành

1. **Config là single source of truth**: không hard-code ngưỡng hay định nghĩa trong code.
2. **Mọi kết quả có lineage**: truy được từ `run_id` về config hash và git commit.
3. **Mọi transformation chỉ fit trên train.**
4. **Time-based split là mặc định**, bắt buộc với dữ liệu panel. Home Credit là ngoại lệ có ghi nhận (không có ngày nộp đơn).
5. **Không commit dữ liệu thô.**
