# Stage 3: Scorecard (Home Credit, champion)

Tổng kết Stage 3 của roadmap: hồi quy logistic trên WoE, kiểm tra dấu hệ số, quy đổi điểm, mã lý do,
model card. Mọi con số trong tài liệu này lấy từ lần chạy chính thức `20260925T021110Z-0531ac69` (commit
`c463d38`, working tree sạch) trên đúng
split / binning / shortlist của Stage 2 (fingerprint `bd8d70cd52bf` / `27c1d5b79454` / `4acf4ed85466`).

## 1. Kết luận nhanh

- **32 biến** trong scorecard: 13 trong 45 biến của shortlist bị loại vì dấu hệ số sai hoặc không ổn định.
- Gini validation **0.5153** (AUC 0.7576, KS 0.388); 45 biến chưa loại cho AUC 0.7582, tức gần như không
  mất gì khi đổi lấy một scorecard giải thích được.
- Điểm nguyên: điểm cơ sở 557, score từ 450 đến 682; làm tròn thay đổi Gini validation +0.0003.
- Bad rate giảm đều qua 10 decile score của validation: 26.5% -> 1.2%.
- PSI của score so với application_test: 0.0013 (Cash loans 0.0023, **Revolving loans 0.1602**).
- Tái lập được: ba lần chạy (hai lần trong lúc phát triển, một lần từ commit sạch `c463d38`) cho cùng
  fingerprint scorecard `a254458deddf`.

## 2. Các bước

| Bước | Làm gì | Đầu ra chính |
|---|---|---|
| 3.1 Fit | Logistic L2 (C = 1) trên WoE của shortlist, chỉ train; loại lần lượt biến có dấu hệ số sai / không ổn định | `scorecard.json`, `sign_trail.csv` |
| 3.2 Điểm | Factor = 20 / ln 2 = 28.85, Offset = 600 - Factor x ln 50 = 487.12; điểm mỗi bin = round(-Factor x b x WoE), điểm cơ sở = round(Offset - Factor x b0) | `scorecard_table.csv` |
| 3.3 Mã lý do | 4 biến làm hồ sơ mất nhiều điểm nhất so với bin tốt nhất của biến đó | `scores.parquet` (`reason_1..4`) |
| 3.4 Model card | Mục đích, dữ liệu, hiệu năng train / validation, biến và điểm, biến bị loại, thuộc tính nhạy cảm, giới hạn | `model_card.md` |

`scores.parquet` có score, PD chưa calibrate và mã lý do cho **mọi** hồ sơ (application_train và
application_test), để chặng 5-6 đọc cùng một đầu ra đã đóng băng thay vì tự chấm điểm lại.

## 3. Quy tắc dấu hệ số

WoE = ln(%Good / %Bad), nên trên P(bad) mọi hệ số phải **âm**. Hệ số dương nghĩa là trong mô hình nhiều
biến, bin rủi ro hơn lại được cộng điểm: thường là biến tương quan khác đã "hút" tín hiệu, và phần còn
lại của biến này đổi chiều. Một scorecard như vậy không giải thích được cho khách hàng.

Mỗi vòng: fit trên toàn bộ train, rồi fit lại 50 lần trên bootstrap phân tầng (resample good và bad
riêng). Biến có hệ số âm trong dưới 95% số lần là không ổn định; biến kém ổn định nhất bị loại, fit
lại, đến khi mọi biến đều ổn định (`model_dev.yaml` -> `champion.sign_stability`).

| Vòng | Biến | Hệ số | Âm trong bootstrap |
|---:|---|---:|---:|
| 1 | `BUR_DAYS_ENDDATE_FACT_MAX` | +0.3185 | 0% |
| 2 | `PREV_COUNT_12M` | +0.2872 | 0% |
| 3 | `PREV_DAYS_DECISION_MIN` | +0.2620 | 0% |
| 4 | `DAYS_BIRTH` | +0.1557 | 0% |
| 5 | `INS_DAYS_LATE_MEAN` | +0.1601 | 2% |
| 6 | `REG_CITY_NOT_WORK_CITY` | +0.0727 | 8% |
| 7 | `CC_PAYMENT_TO_MINIMUM_MEAN` | +0.0985 | 10% |
| 8 | `BUR_ACTIVE_RATE` | +0.0083 | 40% |
| 9 | `REGION_POPULATION_RELATIVE` | +0.0043 | 52% |
| 10 | `INS_AMT_SHORT_SUM` | -0.0301 | 70% |
| 11 | `PREV_APPROVED_RATE` | -0.0833 | 94% |
| 12 | `BUR_ACTIVE_COUNT` | -0.0932 | 94% |
| 13 | `BUR_DAYS_CREDIT_MEAN` | -0.0575 | 94% |

`BUR_DAYS_CREDIT_MEAN` có IV cao thứ 4 (0.1285) nhưng vẫn bị loại ở vòng cuối: khi các biến bureau khác
đã có mặt, phần tín hiệu riêng của nó không đủ ổn định về dấu. AUC validation không đổi sau vòng này.

## 4. Các quyết định ở Stage 3

| Quyết định | Lý do | Ghi ở |
|---|---|---|
| Loại biến theo độ ổn định dấu qua bootstrap, không chỉ theo dấu của một lần fit | Biến có hệ số âm nhỏ nhưng lật dấu trong resample (vòng 10-13) cũng không giải thích được | `model_dev.yaml` -> `champion.sign_stability` |
| C = 1 (L2 nhẹ) | C 1 / 0.01 cho AUC validation như nhau; C 0.001 bắt đầu mất AUC | `model_dev.yaml` -> `champion.C` |
| Điểm làm tròn thành số nguyên, điểm cơ sở tách riêng | Bảng điểm là mô hình người dùng đọc; score là tổng các số nguyên | `models/scorecard.py` |
| Calibration và test không được đọc | Dành cho chặng 5 và 6; có test tự động chứng minh (mục 6) | `tests/test_stage3_dod.py` |

## 5. Thuộc tính nhạy cảm (PROJECT_SCOPE #7)

- `CODE_GENDER` **có** trong scorecard: nữ +3, nam -5 điểm. Chênh 8 điểm = 0.4 PDO, tức odds good:bad
  của nữ cao hơn nam khoảng 1.3 lần khi mọi biến khác như nhau.
- `NAME_FAMILY_STATUS` **có**: từ +2 (goá) đến -1 (độc thân, sống chung).
- `DAYS_BIRTH` **không còn**: bị loại ở vòng 4 vì hệ số dương. Tuổi vẫn có thể đi vào qua biến khác
  (`DAYS_EMPLOYED`, `DAYS_ID_PUBLISH`, `OWN_CAR_AGE`), nên Module 2 vẫn phải phân tích theo nhóm tuổi.

`CODE_GENDER` có thể xuất hiện làm mã lý do (một hồ sơ nam mất 8 điểm so với bin tốt nhất). Đây là hệ quả
trực tiếp của quyết định #7 và là thêm một lý do scorecard này không dùng được ở nơi luật cấm thuộc tính.

## 6. Definition of Done: bằng chứng

`tests/test_stage3_dod.py` chạy chuỗi lệnh thật (ingest -> validate-data -> features -> train):

- cùng dữ liệu cho cùng fingerprint scorecard;
- sửa mọi hồ sơ **calibration và test**: fingerprint scorecard **không đổi**; sửa 300 hồ sơ train: **đổi**
  (đối chứng). Validation và application_test không nằm trong phép thử này vì chúng hợp lệ tham gia
  chọn biến ở Stage 2.5 (thứ tự bin, PSI);
- score, PD và mã lý do trong `scores.parquet` khớp chính xác với việc chấm lại từ `scorecard.json`;
- `scorecard.json` ghi fingerprint của binning và shortlist nó được fit trên; shortlist không khớp binning
  thì bước train dừng.

`tests/test_scorecard.py` kiểm tra quy tắc dấu (biến suppressor bị loại, bản sao nhiễu của một biến không
cùng giữ được dấu ổn định), thang điểm (600 điểm = odds 50:1, +20 điểm = odds gấp đôi), bin rủi ro hơn
không bao giờ được nhiều điểm hơn, và mã lý do.

## 7. Còn mở cho các chặng sau

- **Revolving loans**: PSI score 0.16 so với application_test (mẫu hiện tại chỉ có khoảng 440 hồ sơ loại
  này). Chặng 6 cần báo cáo riêng phân khúc này.
- **Mã lý do cho giá trị missing**: `EXT_SOURCE_1` thiếu ở 56% hồ sơ train. Bin missing được -1 điểm, kém
  bin tốt nhất (+13) tới 14 điểm, nên "EXT_SOURCE_1" là mã lý do thường gặp dù nghĩa thật là "không có
  điểm tín dụng ngoài số 1". Cần câu chữ riêng khi làm dashboard. Tỷ lệ thiếu ở application_test chỉ
  42%: chặng 6 nên xem biến này trong phân tích PSI.
- PD hiện là PD chưa calibrate: chặng 5 fit Platt / isotonic trên calibration sample.

## 8. Tái lập

```
python -m credit_risk run --steps ingest validate-data features train --source home_credit
```

Hoặc, khi đã có đầu ra của Stage 2, chỉ `--steps train` (khoảng 2 phút). Kết quả của mỗi lần chạy nằm ở
`artifacts/<run_id>/train/`.
