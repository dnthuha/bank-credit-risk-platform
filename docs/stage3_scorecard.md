# Stage 3: Scorecard (Home Credit, champion)

Đây là mô hình scorecard được chọn làm champion: mô hình chính để đối chiếu với các mô hình phức tạp hơn.
Chặng này biến WoE thành điểm tín dụng, đồng thời đặt điều kiện để từng biến vẫn có thể giải thích cho
người dùng và đội vận hành. Mọi con số dưới đây lấy từ lần chạy chính thức `20260925T021110Z-0531ac69` (commit
`c463d38`, working tree sạch) trên đúng
split / binning / shortlist của Stage 2 (fingerprint `bd8d70cd52bf` / `27c1d5b79454` / `4acf4ed85466`).

> **Cách đọc nhanh:** AUC/Gini/KS đo khả năng xếp hạng rủi ro; điểm cao hơn tương ứng rủi ro thấp hơn.
> Champion ưu tiên khả năng giải thích và độ ổn định, không chỉ tối đa hóa chỉ số trên một tập validation.

## 1. Kết luận nhanh

- **32 biến** được giữ trong scorecard: 13 trong 45 biến của shortlist bị loại vì dấu hệ số sai hoặc
  không ổn định.
- Gini validation **0.5153** (AUC 0.7576, KS 0.388); 45 biến chưa loại cho AUC 0.7582, tức gần như không
  mất gì khi đổi lấy một scorecard giải thích được.
- Điểm nguyên: điểm cơ sở 557, score từ 450 đến 682; làm tròn thay đổi Gini validation +0.0003.
- Bad rate giảm đều qua 10 decile score của validation: 26.5% -> 1.2%.
- PSI của score so với application_test: 0.0013 (Cash loans 0.0023, **Revolving loans 0.1602**).
- Tái lập được: ba lần chạy (hai lần trong lúc phát triển, một lần từ commit sạch `c463d38`) cho cùng
  fingerprint scorecard `a254458deddf`.

## 2. Quy trình và đầu ra

| Bước | Làm gì | Đầu ra chính |
|---|---|---|
| 3.1 Fit | Logistic L2 (C = 1) trên WoE của shortlist, chỉ train; loại lần lượt biến có dấu hệ số sai / không ổn định | `scorecard.json`, `sign_trail.csv` |
| 3.2 Điểm | Factor = 20 / ln 2 = 28.85, Offset = 600 - Factor x ln 50 = 487.12; điểm mỗi bin = round(-Factor x b x WoE), điểm cơ sở = round(Offset - Factor x b0) | `scorecard_table.csv` |
| 3.3 Mã lý do | 4 biến làm hồ sơ mất nhiều điểm nhất so với bin tốt nhất của biến đó | `scores.parquet` (`reason_1..4`) |
| 3.4 Model card | Mục đích, dữ liệu, hiệu năng train / validation, biến và điểm, biến bị loại, thuộc tính nhạy cảm, giới hạn | `model_card.md` |

`scores.parquet` có score, PD chưa calibrate và mã lý do cho **mọi** hồ sơ (application_train và
application_test), để chặng 5-6 đọc cùng một đầu ra đã đóng băng thay vì tự chấm điểm lại.

## 3. Vì sao phải kiểm tra dấu hệ số

WoE = ln(%Good / %Bad), nên trong mô hình dự báo P(bad), mọi hệ số phải **âm**. Hệ số dương có nghĩa là
một bin rủi ro hơn lại được cộng thêm điểm. Tình huống này thường xảy ra khi biến tương quan khác đã
"hút" phần lớn tín hiệu, khiến phần thông tin còn lại bị đảo chiều. Giữ biến đó sẽ làm scorecard khó giải
thích cho khách hàng và khó bảo vệ trong vận hành.

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

## 4. Chọn biến: tiêu chí và kết quả chi tiết

Phần trên cho thấy **13 biến bị loại theo thứ tự nào**. Phần này bổ sung tiêu chí đầy đủ và kết quả của
cả 45 biến trong shortlist, để người đọc có thể lần từ IV ban đầu đến hệ số, độ ổn định và khoảng điểm
của từng biến được giữ.

### 4.1. Tiêu chí giữ lại

Một biến chỉ được vào scorecard khi đạt **cả ba** tiêu chí sau:

| # | Tiêu chí | Ngưỡng | Kiểm ở |
|---:|---|---|---|
| 1 | Qua shortlist Stage 2.6 | IV train >= 0.02; không có cờ ngoài mẫu của 2.5; \|r\| WoE <= 0.7 với biến IV cao hơn; VIF <= 5 | `features/shortlist.md` |
| 2 | Hệ số trên WoE **âm** | < 0 (WoE = ln(%Good / %Bad): bin nhiều good hơn phải ít rủi ro hơn) | Mô hình fit trên toàn bộ train |
| 3 | Dấu hệ số **ổn định** | âm trong >= 95% của 50 lần fit lại trên bootstrap phân tầng của train | Cùng vòng với tiêu chí 2 |

Tiêu chí 2 và 3 được kiểm theo vòng: mỗi vòng fit lại với các biến còn lại, loại **một** biến kém ổn
định nhất (tỷ lệ âm thấp nhất; nếu hòa, chọn hệ số lớn hơn), rồi fit lại đến khi mọi biến đều đạt. Cách
loại từng biến một là cần thiết vì khi bỏ một biến, dấu của biến tương quan với nó có thể đổi chiều.

### 4.2. Kết quả: 32 biến giữ lại, 13 biến bị loại

| # | Biến | IV train | Hệ số (đủ biến) | Hệ số (cuối / lúc loại) | Âm trong bootstrap | Khoảng điểm | Kết quả | Lý do |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1 | `EXT_SOURCE_2` | 0.3077 | -0.7201 | -0.7181 | 100% | -19 .. 23 | **Giữ** | đạt cả ba tiêu chí |
| 2 | `EXT_SOURCE_3` | 0.3414 | -0.6378 | -0.6526 | 100% | -20 .. 18 | **Giữ** | đạt cả ba tiêu chí |
| 3 | `EXT_SOURCE_1` | 0.1441 | -0.4749 | -0.4550 | 100% | -10 .. 13 | **Giữ** | đạt cả ba tiêu chí |
| 4 | `APP_GOODS_TO_CREDIT` | 0.0757 | -0.4914 | -0.4989 | 100% | -7 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 5 | `CC_UTILISATION_MAX` | 0.0494 | -0.3586 | -0.3569 | 100% | -5 .. 5 | **Giữ** | đạt cả ba tiêu chí |
| 6 | `DAYS_LAST_PHONE_CHANGE` | 0.0448 | -0.2258 | -0.1786 | 100% | -7 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 7 | `INS_LATE_RATE_12M` | 0.0696 | -0.4984 | -0.4883 | 100% | -7 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 8 | `OWN_CAR_AGE` | 0.0240 | -0.6586 | -0.6558 | 100% | -3 .. 7 | **Giữ** | đạt cả ba tiêu chí |
| 9 | `PREV_REFUSED_RATE` | 0.0727 | -0.3162 | -0.3740 | 100% | -6 .. 4 | **Giữ** | đạt cả ba tiêu chí |
| 10 | `BUR_DEBT_TO_CREDIT` | 0.1000 | -0.2957 | -0.3128 | 100% | -5 .. 4 | **Giữ** | đạt cả ba tiêu chí |
| 11 | `INS_AMT_PAID_SUM` | 0.0333 | -0.5756 | -0.5115 | 100% | -4 .. 5 | **Giữ** | đạt cả ba tiêu chí |
| 12 | `PREV_CNT_PAYMENT_MEAN` | 0.0216 | -0.4878 | -0.4883 | 100% | -4 .. 5 | **Giữ** | đạt cả ba tiêu chí |
| 13 | `CC_UTILISATION_MEAN_12M` | 0.0711 | -0.2849 | -0.2475 | 100% | -5 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 14 | `CODE_GENDER` | 0.0381 | -0.7019 | -0.6960 | 100% | -5 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 15 | `DAYS_EMPLOYED` | 0.1143 | -0.3323 | -0.3000 | 100% | -3 .. 5 | **Giữ** | đạt cả ba tiêu chí |
| 16 | `INS_LATE_RATE` | 0.0588 | -0.3513 | -0.3443 | 100% | -4 .. 4 | **Giữ** | đạt cả ba tiêu chí |
| 17 | `NAME_EDUCATION_TYPE` | 0.0521 | -0.4365 | -0.4514 | 100% | -2 .. 6 | **Giữ** | đạt cả ba tiêu chí |
| 18 | `PREV_CREDIT_TO_APPLICATION` | 0.0699 | -0.3719 | -0.3224 | 100% | -5 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 19 | `REGION_RATING_CLIENT_W_CITY` | 0.0481 | -0.2408 | -0.2443 | 100% | -3 .. 4 | **Giữ** | đạt cả ba tiêu chí |
| 20 | `ORGANIZATION_TYPE` | 0.0606 | -0.3453 | -0.2919 | 100% | -2 .. 4 | **Giữ** | đạt cả ba tiêu chí |
| 21 | `POS_CONTRACTS` | 0.0355 | -0.4569 | -0.4209 | 100% | -3 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 22 | `AMT_CREDIT` | 0.0487 | -0.2211 | -0.2173 | 100% | -2 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 23 | `BUR_COUNT_12M` | 0.0872 | -0.1012 | -0.1666 | 100% | -3 .. 2 | **Giữ** | đạt cả ba tiêu chí |
| 24 | `BUR_MAX_OVERDUE` | 0.0232 | -0.4340 | -0.3940 | 100% | -3 .. 2 | **Giữ** | đạt cả ba tiêu chí |
| 25 | `CC_ATM_DRAWINGS_MEAN` | 0.0370 | -0.3138 | -0.2729 | 100% | -4 .. 1 | **Giữ** | đạt cả ba tiêu chí |
| 26 | `DAYS_REGISTRATION` | 0.0280 | -0.2942 | -0.2413 | 100% | -2 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 27 | `FLOORSMAX_MODE` | 0.0365 | -0.2754 | -0.2601 | 100% | -1 .. 4 | **Giữ** | đạt cả ba tiêu chí |
| 28 | `AMT_ANNUITY` | 0.0259 | -0.2600 | -0.2579 | 100% | -2 .. 2 | **Giữ** | đạt cả ba tiêu chí |
| 29 | `DAYS_ID_PUBLISH` | 0.0413 | -0.2584 | -0.2118 | 100% | -2 .. 2 | **Giữ** | đạt cả ba tiêu chí |
| 30 | `FLAG_DOCUMENT_3` | 0.0301 | -0.4061 | -0.3946 | 100% | -1 .. 3 | **Giữ** | đạt cả ba tiêu chí |
| 31 | `OCCUPATION_TYPE` | 0.0780 | -0.1823 | -0.1909 | 100% | -2 .. 2 | **Giữ** | đạt cả ba tiêu chí |
| 32 | `NAME_FAMILY_STATUS` | 0.0202 | -0.2462 | -0.2021 | 100% | -1 .. 2 | **Giữ** | đạt cả ba tiêu chí |
| 33 | `BUR_DAYS_ENDDATE_FACT_MAX` | 0.0439 | +0.3185 | +0.3185 | 0% |  | Loại (vòng 1) | hệ số dương: ngược chiều với chính các bin của nó |
| 34 | `PREV_COUNT_12M` | 0.0204 | +0.2906 | +0.2872 | 0% |  | Loại (vòng 2) | hệ số dương: ngược chiều với chính các bin của nó |
| 35 | `PREV_DAYS_DECISION_MIN` | 0.0436 | +0.2499 | +0.2620 | 0% |  | Loại (vòng 3) | hệ số dương: ngược chiều với chính các bin của nó |
| 36 | `DAYS_BIRTH` | 0.0841 | +0.1611 | +0.1557 | 0% |  | Loại (vòng 4) | hệ số dương: ngược chiều với chính các bin của nó |
| 37 | `INS_DAYS_LATE_MEAN` | 0.0275 | +0.1609 | +0.1601 | 2% |  | Loại (vòng 5) | hệ số dương: ngược chiều với chính các bin của nó |
| 38 | `REG_CITY_NOT_WORK_CITY` | 0.0312 | +0.0647 | +0.0727 | 8% |  | Loại (vòng 6) | hệ số dương: ngược chiều với chính các bin của nó |
| 39 | `CC_PAYMENT_TO_MINIMUM_MEAN` | 0.0226 | +0.0835 | +0.0985 | 10% |  | Loại (vòng 7) | hệ số dương: ngược chiều với chính các bin của nó |
| 40 | `BUR_ACTIVE_RATE` | 0.0942 | -0.0536 | +0.0083 | 40% |  | Loại (vòng 8) | hệ số dương: ngược chiều với chính các bin của nó |
| 41 | `REGION_POPULATION_RELATIVE` | 0.0316 | +0.0020 | +0.0043 | 52% |  | Loại (vòng 9) | hệ số dương: ngược chiều với chính các bin của nó |
| 42 | `INS_AMT_SHORT_SUM` | 0.0416 | -0.0579 | -0.0301 | 70% |  | Loại (vòng 10) | hệ số âm nhưng không ổn định (dưới 95%) |
| 43 | `PREV_APPROVED_RATE` | 0.0665 | -0.1784 | -0.0833 | 94% |  | Loại (vòng 11) | hệ số âm nhưng không ổn định (dưới 95%) |
| 44 | `BUR_ACTIVE_COUNT` | 0.0539 | -0.1066 | -0.0932 | 94% |  | Loại (vòng 12) | hệ số âm nhưng không ổn định (dưới 95%) |
| 45 | `BUR_DAYS_CREDIT_MEAN` | 0.1285 | -0.1803 | -0.0575 | 94% |  | Loại (vòng 13) | hệ số âm nhưng không ổn định (dưới 95%) |

- *Hệ số (đủ biến)* là hệ số ở vòng 1, khi cả shortlist cùng trong mô hình. *Hệ số (cuối / lúc loại)*
  là hệ số của mô hình cuối đối với biến được giữ, hoặc của vòng bị loại đối với biến bị loại. Tỷ lệ âm
  trong bootstrap được đọc theo đúng mô hình tương ứng.
- Biến giữ được xếp theo độ rộng khoảng điểm (biến ảnh hưởng score nhiều nhất lên trước); biến loại được
  xếp theo vòng. Bảng điểm chi tiết theo bin nằm trong `scorecard_table.csv`; lịch sử loại nằm trong
  `sign_trail.csv`.

## 5. Những quyết định quan trọng

| Quyết định | Lý do | Ghi ở |
|---|---|---|
| Loại biến theo độ ổn định dấu qua bootstrap, không chỉ theo dấu của một lần fit | Biến có hệ số âm nhỏ nhưng lật dấu trong resample (vòng 10-13) cũng không giải thích được | `model_dev.yaml` -> `champion.sign_stability` |
| C = 1 (L2 nhẹ) | C 1 / 0.01 cho AUC validation như nhau; C 0.001 bắt đầu mất AUC | `model_dev.yaml` -> `champion.C` |
| Điểm làm tròn thành số nguyên, điểm cơ sở tách riêng | Bảng điểm là mô hình người dùng đọc; score là tổng các số nguyên | `models/scorecard.py` |
| Calibration và test không được đọc | Dành cho chặng 5 và 6; có test tự động chứng minh (mục 7) | `tests/test_stage3_dod.py` |

## 6. Thuộc tính nhạy cảm (PROJECT_SCOPE #7)

- `CODE_GENDER` **có** trong scorecard: nữ +3, nam -5 điểm. Chênh 8 điểm = 0.4 PDO, tức odds good:bad
  của nữ cao hơn nam khoảng 1.3 lần khi mọi biến khác như nhau.
- `NAME_FAMILY_STATUS` **có**: từ +2 (goá) đến -1 (độc thân, sống chung).
- `DAYS_BIRTH` **không còn**: bị loại ở vòng 4 vì hệ số dương. Tuổi vẫn có thể đi vào qua biến khác
  (`DAYS_EMPLOYED`, `DAYS_ID_PUBLISH`, `OWN_CAR_AGE`), nên Module 2 vẫn phải phân tích theo nhóm tuổi.

`CODE_GENDER` có thể xuất hiện làm mã lý do (một hồ sơ nam mất 8 điểm so với bin tốt nhất). Đây là hệ quả
trực tiếp của quyết định #7, và cũng là lý do scorecard này không phù hợp ở nơi pháp luật cấm dùng các
thuộc tính đó.

## 7. Bằng chứng hoàn thành

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

## 8. Việc cần theo dõi ở các chặng sau

- **Revolving loans**: PSI score 0.16 so với application_test (mẫu hiện tại chỉ có khoảng 440 hồ sơ loại
  này). Chặng 6 cần báo cáo riêng phân khúc này.
- **Mã lý do cho giá trị missing**: `EXT_SOURCE_1` thiếu ở 56% hồ sơ train. Bin missing được -1 điểm, kém
  bin tốt nhất (+13) tới 14 điểm, nên "EXT_SOURCE_1" là mã lý do thường gặp dù nghĩa thật là "không có
  điểm tín dụng ngoài số 1". Cần câu chữ riêng khi làm dashboard. Tỷ lệ thiếu ở application_test chỉ
  42%: chặng 6 nên xem biến này trong phân tích PSI.
- PD hiện là PD chưa calibrate: chặng 5 fit Platt / isotonic trên calibration sample.

## 9. Tái lập

```
python -m credit_risk run --steps ingest validate-data features train --source home_credit
```

Hoặc, khi đã có đầu ra của Stage 2, chỉ `--steps train` (khoảng 2 phút). Kết quả của mỗi lần chạy nằm ở
`artifacts/<run_id>/train/`.
