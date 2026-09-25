# Stage 2: WoE, IV và danh sách biến (Home Credit)

Đây là chặng biến dữ liệu thô thành tập biến có thể giải thích cho scorecard. Tài liệu ghi lại
cách dự án kiểm soát leakage, tạo bin, đo sức mạnh từng biến và chọn danh sách biến ban đầu.
Mọi con số dưới đây lấy từ lần chạy chính thức
`20260922T160241Z-dc55d541` (commit `ebb03d0`, working tree sạch).

> **Cách đọc nhanh:** WoE cho biết một nhóm giá trị nghiêng về khách hàng tốt hay xấu; IV đo mức
> phân biệt của biến; PSI đo mức thay đổi phân phối giữa hai tập. Các bin và shortlist chỉ được học
> từ train, rồi mới kiểm tra trên các tập còn lại.

## 1. Kết luận nhanh

- **Hoàn thành tiêu chí Stage 2**: bảng bin tái lập được và chỉ học trên train (mục 5).
- 307,511 hồ sơ train được chia 60 / 10 / 15 / 15; bad rate 8.07% ở cả bốn phần.
- 339 cột được phân loại theo thời điểm có mặt; bảng feature cấp hồ sơ có 192 biến.
- Binning học trên 184,507 hồ sơ train; không biến nào có IV vượt ngưỡng nghi leakage 0.5.
- **45 biến được chọn** cho scorecard; mỗi biến bị loại đều có bước và lý do.

## 2. Quy trình và đầu ra

| Bước | Làm gì | Đầu ra chính | Kết quả |
|---|---|---|---|
| 2.1 Split | Stratified random 60/10/15/15 theo `TARGET`, seed 42 | `splits.parquet` | fingerprint `bd8d70cd52bf` |
| 2.2 Availability matrix | Mỗi cột có mặt từ lúc nào: at_application / historical / post_decision / identifier / target | `configs/feature_availability.yaml` | 339 cột; cột chưa phân loại làm pipeline dừng |
| 2.3 Bảng feature | Gộp 5 bảng lịch sử về cấp hồ sơ bằng DuckDB; "không có lịch sử" là NULL kèm cờ | `features.parquet` | 356,255 hồ sơ (train + application_test), 192 feature |
| 2.4 Binning | 20 pre-bin -> gộp bin < 5% -> ép đơn điệu -> tối đa 8 bin; missing riêng; chỉ học trên train | `binning.json` | fingerprint `27c1d5b79454` |
| 2.5 Kiểm tra ngoài mẫu | Áp nguyên bin sang validation và application_test: IV, thứ tự bin, PSI | `iv_report.md` | 88 biến IV >= 0.02; không biến nào mất quá nửa IV |
| 2.6 Chọn biến | policy -> IV -> cờ 2.5 -> tương quan -> VIF | `shortlist.json` | 45 biến, fingerprint `4acf4ed85466` |

Số biến bị loại ở mỗi tầng của 2.6: IV < 0.02: 104; cờ từ 2.5: 3;
tương quan: 40; VIF: 0; chính sách: 0.

## 3. Những quyết định quan trọng

| Quyết định | Lý do | Ghi ở |
|---|---|---|
| Loại toàn bộ `bureau_balance` | Có lịch sử tháng cho 35.7% khoản bureau ở train nhưng 99.9% ở application_test; mọi `BB_*` có PSI 1.40-1.61 | `feature_availability.yaml` |
| `AMT_CREDIT`, `AMT_ANNUITY` được phép chữ U ngược (một lần đổi chiều) | Hình dạng giữ nguyên khi chỉ xét Cash loans; khoản lớn chỉ duyệt cho khách tốt hơn | `model_dev.yaml` -> `allow_non_monotonic` |
| `AMT_GOODS_PRICE` giữ đơn điệu, rồi bị loại vì trùng `AMT_CREDIT` (Spearman 0.98) | "Đỉnh" rủi ro là một giá trị tròn (450,000), không phải quan hệ giá hàng - rủi ro | `model_dev.yaml` |
| Không loại thuộc tính nhạy cảm theo chính sách | Quyết định của chủ project | PROJECT_SCOPE #7 |
| Tổng / trung bình số thực tính bằng DECIMAL chính xác | Phép cộng số thực phụ thuộc thứ tự dòng: sau khi ingest lại, một hồ sơ nhảy bin | `features/build.py` |

**Lưu ý về thuộc tính nhạy cảm:** `CODE_GENDER`, `NAME_FAMILY_STATUS` và `DAYS_BIRTH` vẫn nằm trong
danh sách 45 biến theo quyết định phạm vi dự án. Vì vậy, scorecard có thể cho điểm khác nhau theo giới
tính (nam WoE -0.25) và tình trạng hôn nhân (độc thân -0.21). Đây không phải chi tiết kỹ thuật nhỏ:
model card ở Stage 3 phải nêu rủi ro này, và Module 2 phải báo cáo hiệu năng theo ba nhóm
(`validation.yaml`, `fairness_review`).

## 4. Danh sách 45 biến

Theo nhóm: Hồ sơ hiện tại 19, Bureau 7, Hồ sơ vay cũ 6, Lịch trả nợ 5, Thẻ tín dụng 4, Điểm ngoài 3, POS / cash 1.

| # | Biến | Nhóm | Xu hướng bad rate | IV train | IV validation | PSI vs application_test |
|---:|---|---|---|---:|---:|---:|
| 1 | `EXT_SOURCE_3` | Điểm ngoài | giảm | 0.3414 | 0.2938 | 0.0073 |
| 2 | `EXT_SOURCE_2` | Điểm ngoài | giảm | 0.3077 | 0.2896 | 0.0156 |
| 3 | `EXT_SOURCE_1` | Điểm ngoài | giảm | 0.1441 | 0.1691 | 0.0813 |
| 4 | `BUR_DAYS_CREDIT_MEAN` | Bureau | tăng | 0.1285 | 0.1161 | 0.0084 |
| 5 | `DAYS_EMPLOYED` | Hồ sơ hiện tại | tăng | 0.1143 | 0.1393 | 0.0051 |
| 6 | `BUR_DEBT_TO_CREDIT` | Bureau | tăng | 0.1000 | 0.1145 | 0.0046 |
| 7 | `BUR_ACTIVE_RATE` | Bureau | tăng | 0.0942 | 0.0995 | 0.0050 |
| 8 | `BUR_COUNT_12M` | Bureau | tăng | 0.0872 | 0.0902 | 0.0041 |
| 9 | `DAYS_BIRTH` | Hồ sơ hiện tại | tăng | 0.0841 | 0.0966 | 0.0006 |
| 10 | `OCCUPATION_TYPE` | Hồ sơ hiện tại | theo bad rate | 0.0780 | 0.0755 | 0.0010 |
| 11 | `APP_GOODS_TO_CREDIT` | Hồ sơ hiện tại | giảm | 0.0757 | 0.0677 | 0.0344 |
| 12 | `PREV_REFUSED_RATE` | Hồ sơ vay cũ | tăng | 0.0727 | 0.0833 | 0.0416 |
| 13 | `CC_UTILISATION_MEAN_12M` | Thẻ tín dụng | tăng | 0.0711 | 0.0474 | 0.0127 |
| 14 | `PREV_CREDIT_TO_APPLICATION` | Hồ sơ vay cũ | tăng | 0.0699 | 0.0938 | 0.0377 |
| 15 | `INS_LATE_RATE_12M` | Lịch trả nợ | tăng | 0.0696 | 0.0653 | 0.0027 |
| 16 | `PREV_APPROVED_RATE` | Hồ sơ vay cũ | giảm | 0.0665 | 0.0816 | 0.0474 |
| 17 | `ORGANIZATION_TYPE` | Hồ sơ hiện tại | theo bad rate | 0.0606 | 0.0597 | 0.0010 |
| 18 | `INS_LATE_RATE` | Lịch trả nợ | tăng | 0.0588 | 0.0636 | 0.0465 |
| 19 | `BUR_ACTIVE_COUNT` | Bureau | tăng | 0.0539 | 0.0624 | 0.0028 |
| 20 | `NAME_EDUCATION_TYPE` | Hồ sơ hiện tại | theo bad rate | 0.0521 | 0.0527 | 0.0009 |
| 21 | `CC_UTILISATION_MAX` | Thẻ tín dụng | tăng | 0.0494 | 0.0339 | 0.0202 |
| 22 | `AMT_CREDIT` | Hồ sơ hiện tại | chữ U ngược | 0.0487 | 0.0567 | 0.0854 |
| 23 | `REGION_RATING_CLIENT_W_CITY` | Hồ sơ hiện tại | tăng | 0.0481 | 0.0620 | 0.0022 |
| 24 | `DAYS_LAST_PHONE_CHANGE` | Hồ sơ hiện tại | tăng | 0.0448 | 0.0499 | 0.0334 |
| 25 | `BUR_DAYS_ENDDATE_FACT_MAX` | Bureau | tăng | 0.0439 | 0.0452 | 0.0050 |
| 26 | `PREV_DAYS_DECISION_MIN` | Hồ sơ vay cũ | tăng | 0.0436 | 0.0446 | 0.0626 |
| 27 | `INS_AMT_SHORT_SUM` | Lịch trả nợ | tăng | 0.0416 | 0.0378 | 0.0458 |
| 28 | `DAYS_ID_PUBLISH` | Hồ sơ hiện tại | tăng | 0.0413 | 0.0373 | 0.0502 |
| 29 | `CODE_GENDER` | Hồ sơ hiện tại | theo bad rate | 0.0381 | 0.0423 | 0.0007 |
| 30 | `CC_ATM_DRAWINGS_MEAN` | Thẻ tín dụng | tăng | 0.0370 | 0.0259 | 0.0052 |
| 31 | `FLOORSMAX_MODE` | Hồ sơ hiện tại | giảm | 0.0365 | 0.0429 | 0.0028 |
| 32 | `POS_CONTRACTS` | POS / cash | giảm | 0.0355 | 0.0223 | 0.0457 |
| 33 | `INS_AMT_PAID_SUM` | Lịch trả nợ | giảm | 0.0333 | 0.0260 | 0.0477 |
| 34 | `REGION_POPULATION_RELATIVE` | Hồ sơ hiện tại | giảm | 0.0316 | 0.0404 | 0.0027 |
| 35 | `REG_CITY_NOT_WORK_CITY` | Hồ sơ hiện tại | tăng | 0.0312 | 0.0251 | 0.0002 |
| 36 | `FLAG_DOCUMENT_3` | Hồ sơ hiện tại | tăng | 0.0301 | 0.0248 | 0.0319 |
| 37 | `DAYS_REGISTRATION` | Hồ sơ hiện tại | tăng | 0.0280 | 0.0230 | 0.0007 |
| 38 | `INS_DAYS_LATE_MEAN` | Lịch trả nợ | tăng | 0.0275 | 0.0303 | 0.0420 |
| 39 | `AMT_ANNUITY` | Hồ sơ hiện tại | chữ U ngược | 0.0259 | 0.0351 | 0.0337 |
| 40 | `OWN_CAR_AGE` | Hồ sơ hiện tại | tăng | 0.0240 | 0.0145 | 0.0004 |
| 41 | `BUR_MAX_OVERDUE` | Bureau | tăng | 0.0232 | 0.0220 | 0.0058 |
| 42 | `CC_PAYMENT_TO_MINIMUM_MEAN` | Thẻ tín dụng | giảm | 0.0226 | 0.0212 | 0.0043 |
| 43 | `PREV_CNT_PAYMENT_MEAN` | Hồ sơ vay cũ | tăng | 0.0216 | 0.0343 | 0.0336 |
| 44 | `PREV_COUNT_12M` | Hồ sơ vay cũ | tăng | 0.0204 | 0.0401 | 0.0419 |
| 45 | `NAME_FAMILY_STATUS` | Hồ sơ hiện tại | theo bad rate | 0.0202 | 0.0167 | 0.0030 |

"Xu hướng" là chiều thay đổi của bad rate khi giá trị tăng; với biến categorical, các nhóm được xếp theo
bad rate. IV ở validation dùng nguyên các bin đã học từ train. Vì validation chỉ có 30,750 hồ sơ, IV của
biến yếu có thể cao hơn train do nhiễu mẫu; điều đó không tự động chứng minh biến mạnh hơn ngoài mẫu.

## 5. Bằng chứng hoàn thành

Tiêu chí của overview: **"Bảng bin tái lập được và chỉ học trên train."**

**Binning chỉ học từ train.** `tests/test_stage2_dod.py` chạy chuỗi lệnh thật
(`ingest -> validate-data -> features`) và xác nhận:

- sửa mọi hồ sơ ngoài train (validation, calibration, test) và toàn bộ application_test: fingerprint bảng bin **không đổi**;
- sửa đúng như vậy trên 5 hồ sơ train: fingerprint **đổi** (đối chứng: phép kiểm tra đủ nhạy).

**Kết quả tái lập được.** Hai lần chạy toàn bộ pipeline từ commit `ebb03d0`, mỗi lần ingest lại từ đầu,
cho cùng các fingerprint sau:

| Run | Split | Binning | Shortlist |
|---|---|---|---|
| `20260922T155938Z-dc55d541` | `bd8d70cd52bf` | `27c1d5b79454` | `4acf4ed85466` |
| `20260922T160241Z-dc55d541` | `bd8d70cd52bf` | `27c1d5b79454` | `4acf4ed85466` |

Hai lỗi tái lập đã được phát hiện và sửa trên đường đi, cả hai đều do phép cộng số thực phụ thuộc thứ
tự: lần đầu trong một lần chạy (DuckDB cộng song song), lần hai qua các lần ingest (thứ tự dòng đổi).
Cách sửa cuối cùng là cộng bằng DECIMAL chính xác, có test xáo trộn thứ tự dòng và đổi số luồng.

## 6. Bàn giao cho Stage 3

Stage 3 (scorecard) đọc `data/processed/home_credit/binning.json` và `shortlist.json`. Các việc còn lại
ở chặng kế tiếp là:

- hồi quy logistic trên WoE của 45 biến, kiểm tra dấu hệ số (`require_stable_coefficient_sign`);
  biến có dấu ngược hoặc không ổn định bị loại tiếp;
- quy đổi điểm theo `definitions.yaml` (BaseScore 600, BaseOdds 50, PDO 20), mã lý do, model card;
- model card nêu các giới hạn: không có out-of-time thật, population application_test khác train, và
  rủi ro fairness của quyết định #7.

## 7. Tái lập

```
python -m credit_risk run --steps ingest validate-data features --source home_credit
```

Kết quả của mỗi lần chạy nằm ở `artifacts/<run_id>/features/` (không commit lên git).
