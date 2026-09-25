# bank-credit-risk-platform

Nền tảng bao quát toàn bộ **credit risk model lifecycle**: từ một PD scorecard và một machine learning challenger, đến independent validation, portfolio monitoring và Early Warning System cho các khoản vay có nguy cơ chuyển sang 90+ DPD.

```
Model Development  ->  Independent Model Validation  ->  Portfolio Monitoring  ->  Early Warning
   (Home Credit)            (Home Credit)                 (Freddie Mac)            (Freddie Mac)
```

- Phạm vi, quyết định đã chốt và các giới hạn: [PROJECT_SCOPE.md](PROJECT_SCOPE.md)
- Cách tải dữ liệu: [data/README.md](data/README.md)
- Chất lượng dữ liệu: [Home Credit](docs/data_quality_home_credit.md), [Freddie Mac](docs/data_quality_freddie_mac.md)
- Stage 2, WoE / IV và danh sách biến: [docs/stage2_woe_iv.md](docs/stage2_woe_iv.md)
- Stage 3, scorecard: [docs/stage3_scorecard.md](docs/stage3_scorecard.md)
- Stage 4, LightGBM challenger: [docs/stage4_lightgbm.md](docs/stage4_lightgbm.md)
- Định nghĩa nghiệp vụ (single source of truth): [configs/definitions.yaml](configs/definitions.yaml)

## Trạng thái

| Chặng | Nội dung | Trạng thái |
|---|---|---|
| 1 | Nền móng: repo, môi trường, data contract, định nghĩa, ingest, data validation, test lõi | Dữ liệu thật đã qua validate: Home Credit (135 check, 0 error), Freddie Mac 2012-2025 (0 error) |
| 2 | WoE / IV, binning, rà soát leakage | **Xong** (tag `v0.2-woe-iv`): 45 biến, bảng bin tái lập được và chỉ học trên train. [Tổng kết](docs/stage2_woe_iv.md) |
| 3 | Scorecard | **Xong**: 32 biến, điểm nguyên (BaseScore 600, odds 50:1, PDO 20), Gini validation 0.515, mã lý do, model card. [Tổng kết](docs/stage3_scorecard.md) |
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
| 3 | `features` | Split, availability matrix, bảng feature cấp hồ sơ, binning + WoE fit trên train, báo cáo IV, shortlist (M1); panel + biến trễ (M3) còn lại | `data/processed/home_credit/{splits,features}.parquet`, `binning.json`, `shortlist.json`, `artifacts/<run_id>/features/` |
| 4 | `train` | Scorecard: logistic trên WoE, loại biến sai / không ổn định dấu, quy đổi điểm, mã lý do, model card; LightGBM challenger: 30 cấu hình, early stopping, SHAP, so với champion (M1) | `data/processed/home_credit/{scorecard.json,scores.parquet,challenger.json,challenger_model.txt,challenger_scores.parquet}`, `artifacts/<run_id>/train/` |
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
│   ├── feature_availability.yaml # mỗi cột có mặt từ lúc nào: control chống leakage
│   └── data_contracts/           # data dictionary + rule kiểm tra cho từng nguồn
├── data/README.md                # chỉ hướng dẫn tải
├── docs/                         # data quality, phát hiện từ dữ liệu thật
├── src/credit_risk/
│   ├── data/                     # contracts, ingest, validate (DuckDB)
│   ├── features/                 # split.py, availability.py, build.py, binning.py, iv_report.py, selection.py, woe.py
│   ├── models/                   # scorecard.py (champion), challenger.py (LightGBM, SHAP)
│   ├── validation/               # metrics.py: AUC, Gini, KS, Brier, PSI
│   ├── monitoring/               # buckets.py, transitions.py (roll rate)
│   ├── ews/                      # labels.py: nhãn 90+ DPD trong 3 tháng
│   ├── reporting/                # model_card.py, challenger_report.py
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
