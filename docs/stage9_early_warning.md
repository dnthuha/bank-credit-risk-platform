# Stage 9: Survival và Early Warning System (Freddie Mac)

Chặng này trả lời câu hỏi thứ hai của Module 3: **khoản nào sắp vào 90+ DPD?** Một lệnh `ews` gán nhãn
3 tháng trên toàn bộ panel, dựng feature tại từng tháng quan sát, huấn luyện hai mô hình, chấm điểm mọi đoạn
thời gian, rồi lập **watchlist** cho tháng mới nhất theo công suất xử lý của đội vận hành. Cùng bước đó chạy
survival analysis mô tả theo tuổi khoản vay (MOB), với trả nợ trước hạn là competing risk.

Con số dưới đây lấy từ lần chạy `20261025T151857Z-91261e20`: **52 kiểm tra, 0 lệch**. Toàn bộ bước chạy
khoảng 8.5 phút trên laptop 7.4 GB RAM.

> **Cách đọc nhanh:** watchlist mỗi tháng là 5% khoản có PD cao nhất. Ở out-of-time 2022-2025, danh sách
> đó chứa **90%** các khoản vào 90+ trong 3 tháng sau. Phần lớn là khoản **đã** trễ 30 / 60 ngày, nhóm mà
> mô hình nào cũng bắt được. Với khoản **đang trả đúng hạn**, danh sách bắt được khoảng **một nửa**. Đó mới
> là phần cảnh báo sớm thực sự.

## 1. Kết luận nhanh

- **Watchlist bắt được 90.3% sự kiện ở out-of-time** (2022-01 → 2025-12, 13.3 triệu tháng-khoản vay, 32,777
  sự kiện, tỷ lệ 0.247%), với lift 18 và precision 4.5%. ROC AUC 0.971, PR AUC 0.391 (mốc ngẫu nhiên là 0.0025).
  Capture ổn định qua từng năm: 88.6% (2022) → 91.7% (2025).
- **Khoản đang current tại t là phần khó:** chúng chiếm 19.9% sự kiện out-of-time, và danh sách bắt được
  51.5% trong số đó (ROC AUC trong nhóm 0.87). Khoản đang 30 / 60 ngày quá hạn luôn có mặt trong danh sách
  (capture 100%).
- **COVID là điểm yếu rõ nhất:** ở stress test 2020-03 → 2021-09, capture giảm còn 77.5%, và chỉ 19.5% với
  khoản current. Forbearance làm khoản đang trả bình thường chuyển thẳng sang 90+, điều mà lịch sử trả nợ không
  báo trước.
- **Hai mô hình gần như ngang nhau.** LightGBM được chọn theo PR AUC trên validation (0.329 so với 0.318),
  nhưng ở out-of-time logistic baseline nhỉnh hơn (0.408 so với 0.391; capture 90.7% so với 90.3%). Tín hiệu
  gần như nằm hết trong vài feature hành vi (quá hạn trong 6 tháng, thay đổi bucket, thời gian từ lần trễ gần
  nhất); mô hình phức tạp hơn không thêm được bao nhiêu.
- **PD không khớp mức thực tế ở mọi đoạn**, đúng như quyết định không hiệu chỉnh lại: cao hơn thực tế ở
  validation và out-of-time (PD trung bình 0.322% so với 0.247%), thấp hơn ở stress (0.738% so với 0.921%).
  Watchlist xếp hạng theo điểm nên không bị ảnh hưởng.
- **Watchlist 2026-03:** 16,159 khoản (5% của 323,177 khoản trong cohort), dư nợ 4.6 tỷ USD, PD trung bình
  7.3%. 12,335 khoản đang current, 2,977 đang trễ 30 ngày, 847 đang trễ 60 ngày. Theo PD, kỳ vọng khoảng
  1,179 khoản vào 90+ trong 3 tháng tới.
- **Survival:** sau 60 tháng, 3.3% khoản đã default và 46.8% đã trả hết (Aalen-Johansen). Nếu không ai trả
  trước, xác suất default sẽ là 4.5% (Kaplan-Meier). FICO dưới 620 default gấp khoảng 10 lần FICO 780+ ở mọi MOB.

## 2. Quyết định (2026-10-25)

| # | Vấn đề | Quyết định | Ghi ở |
|---|---|---|---|
| 1 | Train 10 triệu dòng, 0.12% sự kiện, laptop 7.4 GB | **Giữ mọi event + 5% non-event**, non-event mang trọng số 20; validation / stress / out-of-time chấm điểm đầy đủ | `ews.yaml -> ews.train_sample` |
| 2 | Survival làm tới đâu | **Mô tả, tự viết bằng numpy**: Kaplan-Meier, Aalen-Johansen, delayed entry; không thêm thư viện | `ews.survival` |
| 3 | Tỷ lệ sự kiện đổi mạnh giữa các đoạn | **Báo cáo, không hiệu chỉnh lại PD**; calibration riêng từng đoạn | `ews.yaml` (ghi chú), mục 4.4 |
| 4 | Dùng cờ hỗ trợ của servicer | **Dùng**: forbearance, deferral, modification, disaster | `ews.features` |

Mẫu train lấy bằng hash của `(loan_id, period, seed)` trong DuckDB, nên không phụ thuộc thứ tự dòng: 509,825
dòng, gồm 11,542 event. Tỷ lệ sự kiện có trọng số của mẫu khớp tỷ lệ của toàn đoạn train trong 10% (một kiểm tra).

## 3. Thứ tự và kiểm tra

```
9.0 cohort     panel (8.0) -> ews_labels.parquet, split theo thời gian     12 kiểm tra (+10 của panel)
9.1 features   availability matrix, feature tại t -> ews_dataset.parquet   13
9.2 models     baseline + challenger, chấm mọi dòng -> ews_scores.parquet   6
9.3 watchlist  top 5% tháng chấm điểm, reason code                         5
9.4 survival   KM / AJ theo MOB, đối chiếu vintage curve chặng 8           6
```

Panel, nhãn và feature có kiểm tra lệch thì bước dừng: không con số nào dựng trên chúng đáng tin nữa.

**Nhãn (9.0).** Định nghĩa trong `ews/labels.py` (pandas, có unit test từng quy tắc) không chạy được trên 35.6
triệu dòng, nên `ews/cohort.py` viết lại bằng SQL trên monitoring panel. Mỗi lần chạy lấy mẫu 20,000 khoản (có
seed), gán nhãn lại bằng bản pandas từ file performance gốc và đòi khớp **tuyệt đối** cả năm cột (bucket tại t,
cờ từng 90+, y, tháng xảy ra sự kiện, lý do loại trừ): 1,022,681 dòng, 0 khác biệt. Thêm vào đó: mọi y = 1 trỏ
đúng vào một tháng 90+ / thanh lý trong panel, không dòng y = 0 nào có tháng xấu trong cửa sổ, và mọi dòng thuộc
một đoạn đều có đủ 3 tháng dữ liệu phía sau.

Mỗi đoạn phải kết thúc ít nhất 3 tháng trước khi đoạn sau bắt đầu (config bị từ chối nếu không), và mọi đoạn
dừng ở tháng cuối còn đủ cửa sổ nhãn.

**Feature (9.1).** Mỗi feature khai báo các cột gốc nó đọc. Ma trận availability của Freddie Mac (mới, trong
`feature_availability.yaml`) phân loại **mọi** cột của hai bảng: lúc giải ngân, trạng thái tháng t (historical),
hay chỉ có sau khi thoát (post_decision, ví dụ các cột tổn thất, `zero_balance_code`). Feature đọc một cột không
`use: true` thì bước dừng; một cột mới chưa phân loại cũng làm bước dừng. Window frame kết thúc ở `CURRENT ROW`,
và một test sửa mọi tháng sau t (bucket, dư nợ, lãi suất, cờ hỗ trợ) để chứng minh feature tại t không đổi.

**Survival (9.4).** Trên static pool (khoản có lịch sử từ đầu), CIF default của Aalen-Johansen phải lớn hơn hoặc
bằng vintage curve chặng 8 và lệch không quá 2% tương đối: censoring chỉ có thể làm ước lượng tăng. Thực tế lệch
tối đa 0.011 điểm phần trăm (0.37% tương đối) trên 45 cặp lứa × MOB.

## 4. Kết quả

### 4.1 Nhãn và cohort (9.0)

| Đoạn | Tháng | Cohort | Sự kiện | Tỷ lệ | Khoản |
|---|---|---:|---:|---:|---:|
| train | 2014-01 → 2018-09 | 10,020,108 | 11,542 | 0.115% | 325,330 |
| validation | 2019-01 → 2019-09 | 2,385,091 | 3,035 | 0.127% | 294,068 |
| stress test | 2020-03 → 2021-09 | 4,521,562 | 41,622 | 0.921% | 351,991 |
| out-of-time | 2022-01 → 2025-12 | 13,290,646 | 32,777 | 0.247% | 420,533 |

Khớp ước tính sơ bộ trong tài liệu chất lượng dữ liệu (mục 7), chỉ lệch 9 sự kiện ở out-of-time. Dòng bị loại
khỏi cohort chủ yếu vì khoản đã thoát (inactive) hoặc đã 90+ sẵn tại t; cửa sổ thiếu tháng chỉ chiếm vài nghìn dòng
mỗi đoạn.

### 4.2 Feature và mô hình (9.2)

34 feature: 10 lúc giải ngân (FICO, LTV, CLTV, DTI, lãi suất, dư nợ, kỳ hạn, MI, số người vay, số căn), 16 hành
vi tại t và trong cửa sổ 6 tháng, 8 phân loại (gồm bang và kế hoạch hỗ trợ tại t). DTI thiếu ở 20% dòng train,
phần lớn là khoản HARP (thiếu có cấu trúc, có bin riêng).

- **Baseline:** logistic trên feature chia bin theo quantile có trọng số. Một giá trị chiếm từ một nửa mẫu trở
  lên (999 = chưa từng trễ, 0 = không deferral) có bin riêng; feature ít giá trị (bucket 0/1/2) có một bin cho mỗi
  giá trị; missing là một bin.
- **Challenger:** LightGBM, 3 cấu hình, early stopping theo average precision trên validation. Cấu hình
  `num_leaves 63, learning_rate 0.03, min_child_samples 500` thắng (323 cây). Hai cấu hình nhỏ hơn dừng sau 4
  cây với PR AUC 0.29-0.30.
- **Feature quan trọng nhất** (gain của challenger): `max_bucket_window` 48.5%, `bucket_change_1m` 15.6%,
  `months_since_delinquent` 7.8%, `property_state` 5.0%, `bucket_num` 4.2%, `months_30plus_window` 4.1%.
  Thuộc tính lúc giải ngân (FICO, CLTV, dư nợ) mỗi cái dưới 2.5%.

| Đoạn | Mô hình | PR AUC | ROC AUC | Capture top 5% | Precision top 5% | Lift | Capture top 10% |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | baseline | 0.318 | 0.972 | 92.3% | 2.3% | 18.5 | 93.7% |
| validation | **challenger** | 0.329 | 0.972 | 91.9% | 2.3% | 18.4 | 93.5% |
| stress test | baseline | 0.538 | 0.921 | 77.6% | 14.3% | 15.5 | 81.5% |
| stress test | challenger | 0.535 | 0.921 | 77.5% | 14.3% | 15.5 | 80.7% |
| out-of-time | baseline | 0.408 | 0.971 | 90.7% | 4.5% | 18.1 | 93.1% |
| out-of-time | challenger | 0.391 | 0.971 | 90.3% | 4.5% | 18.1 | 92.7% |

Capture / precision / lift ở top k% được tính **trong từng tháng** rồi cộng lại, đúng như cách watchlist được
dựng mỗi tháng. PR AUC phải đọc so với tỷ lệ sự kiện của đoạn (0.13% validation, 0.92% stress, 0.25% out-of-time).

Quy tắc chọn mô hình theo validation là quy tắc đặt trước. Kết quả out-of-time ngược chiều nhưng chênh lệch
rất nhỏ (capture 0.4 điểm), nên chưa đủ lý do đổi quy tắc. Chặng thẩm định sau nên xét việc dùng baseline,
vì nó đơn giản hơn và dễ giải thích hơn.

### 4.3 Backtest watchlist

| Đoạn | Năm | Sự kiện | Bắt được | Capture | Precision | Lift |
|---|---|---:|---:|---:|---:|---:|
| validation | 2019 | 3,035 | 2,788 | 91.9% | 2.34% | 18.4 |
| stress test | 2020 | 35,954 | 27,254 | 75.8% | 21.67% | 15.2 |
| stress test | 2021 | 5,668 | 4,999 | 88.2% | 4.98% | 17.6 |
| out-of-time | 2022 | 5,921 | 5,245 | 88.6% | 3.79% | 17.7 |
| out-of-time | 2023 | 6,779 | 6,080 | 89.7% | 3.85% | 17.9 |
| out-of-time | 2024 | 9,525 | 8,605 | 90.3% | 4.86% | 18.1 |
| out-of-time | 2025 | 10,552 | 9,678 | 91.7% | 5.06% | 18.3 |

Theo trạng thái tại t, ở out-of-time:

| Trạng thái tại t | Dòng | Sự kiện | Tỷ lệ | Phần của danh sách | Capture |
|---|---:|---:|---:|---:|---:|
| current | 13,147,342 | 6,532 | 0.050% | 78.4% | 51.5% |
| dpd30 | 114,874 | 11,836 | 10.3% | 17.3% | 100% |
| dpd60 | 28,430 | 14,409 | 50.7% | 4.3% | 100% |

Danh sách đủ chỗ cho mọi khoản đang trễ (chưa tới 1% cohort), và phần còn lại dành cho khoản current có rủi ro
cao nhất. Capture theo FICO giảm dần từ 93% (FICO thấp) xuống 84% (780+), vì sự kiện ở nhóm FICO cao ít báo trước hơn.

### 4.4 Calibration

| Đoạn | Mô hình | PD trung bình | Thực tế | Thực tế / PD |
|---|---|---:|---:|---:|
| validation | baseline | 0.161% | 0.127% | 0.79 |
| validation | challenger | 0.196% | 0.127% | 0.65 |
| stress test | baseline | 0.655% | 0.921% | 1.40 |
| stress test | challenger | 0.738% | 0.921% | 1.25 |
| out-of-time | baseline | 0.273% | 0.247% | 0.90 |
| out-of-time | challenger | 0.322% | 0.247% | 0.77 |

Theo decile ở out-of-time, chín decile đầu thực tế cao hơn PD (khoảng gấp 1.5-2 lần, nhưng đều dưới 0.07%), còn
decile cao nhất PD 3.1% so với thực tế 2.3%. Nếu dùng PD cho mục đích khác ngoài xếp hạng (ví dụ dự phòng), cần
hiệu chỉnh lại trên dữ liệu gần nhất. Việc đó nằm ngoài phạm vi chặng này.

### 4.5 Watchlist 2026-03 (9.3)

Danh sách 16,159 khoản, sắp theo PD giảm dần. Mỗi khoản có 3 reason code, là các feature đẩy log-odds lên nhiều
nhất so với một khoản trung bình (TreeSHAP với LightGBM; với logistic là hệ số của bin trừ trung bình có trọng số
trên train), viết dạng `feature=giá trị`, ví dụ `max_bucket_window=2 | months_since_delinquent=0 | bucket_change_1m=1`.

Reason code xuất hiện nhiều nhất: `months_since_delinquent` (11,112 lần), `paydown_window` (7,300),
`property_state` (6,607), `months_30plus_window` (6,579), `fico` (6,062). Với 12,335 khoản đang current, lý do
thường là đã từng trễ gần đây, dư nợ không giảm trong 6 tháng (không trả gốc), bang rủi ro cao và FICO thấp.

### 4.6 Survival (9.4)

699,995 khoản, trong đó 20,416 khoản bị cắt trái vào risk set từ MOB đầu tiên có dữ liệu (delayed entry). Kết
cục: 24,171 default, 361,217 trả hết, 314,607 censored (còn hoạt động hoặc thoát không rõ kết cục).

| MOB | 12 | 24 | 36 | 60 |
|---|---:|---:|---:|---:|
| CIF default (Aalen-Johansen) | 0.56% | 1.41% | 2.19% | 3.31% |
| KM default (payoff coi là censored) | 0.58% | 1.57% | 2.62% | 4.46% |

Khoảng cách giữa hai dòng tăng theo MOB vì trả trước loại dần khoản khỏi rủi ro: đến MOB 60, 46.8% đã trả hết.
KM trả lời câu hỏi "nếu không ai trả trước"; CIF trả lời câu hỏi "bao nhiêu phần danh mục thực sự default".

CIF default MOB 60 theo FICO: <620 10.9%, 620-679 9.7%, 680-739 5.1%, 740-779 2.4%, 780+ 1.0%. Theo LTV, tăng
từ 2.1% (≤60) lên 7.6% (96-100). Theo lứa, lứa 2017-2019 cao nhất (5.1-5.3% ở MOB 60) vì đi qua COVID ở MOB
24-36; lứa 2022-2024 ở MOB 24 khoảng 1.8%, gấp 3 lần lứa 2012-2016, khớp cảnh báo đỏ của chặng 8.

Bỏ khoản bị cắt trái ra thì đường cong gần như không đổi (CIF MOB 60: 3.33% so với 3.31%): delayed entry đúng
quy cách, và 2.9% khoản bị cắt trái không làm lệch kết quả.

## 5. Đầu ra

`data/processed/freddie_mac/`:

| File | Nội dung |
|---|---|
| `ews_labels.parquet` | Một dòng mỗi tháng-khoản vay: đoạn, regime, y, tháng sự kiện, lý do loại trừ (16 MB) |
| `ews_dataset.parquet` | Cohort của bốn đoạn và dòng chấm điểm, cùng 34 feature (825 MB) |
| `ews_scores.parquet` | PD của cả hai mô hình cho mọi dòng trên (774 MB) |

`artifacts/<run_id>/ews/`: `ews_metrics.json` (provenance, kiểm tra, giới hạn, các bảng), `ews_report.md`
(kết quả chính và watchlist trước, rồi giới hạn, kiểm tra, từng module), `watchlist.csv`, `by_split.csv`,
`monthly_lists.csv`, `calibration.csv`, `segments.csv`, `survival_curves.csv`, `trials.csv`, `importance.csv`,
`coefficients.csv`, `feature_columns.csv`, `availability.csv`, `checks.csv`, và `models/`: `challenger.txt`
(LightGBM), `baseline.json` (hệ số), `encoder.json` (bin và mức phân loại).

Fingerprint của lần chạy này: baseline `58f9b7e36631`, challenger `960f590e1dd5`.

## 6. Definition of Done: bằng chứng

| Yêu cầu | Bằng chứng |
|---|---|
| Nhãn đúng định nghĩa, cả trên dữ liệu lớn | `test_labels.py` (mỗi quy tắc một test); SQL = pandas trên mọi dòng (`test_ews_cohort.py`) và trên 20,000 khoản thật mỗi lần chạy |
| Không leakage theo thời gian | Khoảng cách giữa các đoạn được kiểm khi đọc config; feature tại t không đổi khi sửa tháng sau t (`test_ews_features.py`); cột không được phép làm bước dừng (`test_stage9_dod.py`) |
| Một lệnh, tái lập được | `test_stage9_dod.py`: chạy hai lần cho cùng `ews_metrics.json` và cùng `watchlist.csv`; LightGBM deterministic, mẫu train lấy bằng hash |
| Watchlist theo config | Công suất 10% và tháng chấm điểm khác cho đúng kích thước và tháng (`test_stage9_dod.py`) |
| Chỉ số và survival đúng | `test_ews_metrics.py` (average precision khớp sklearn, danh sách tháng, tie), `test_ews_survival.py` (ví dụ tính tay có delayed entry, không censoring thì CIF = tỷ lệ quan sát) |
| Mô hình và reason code | `test_ews_models.py`: bin, one-hot, đóng góp của logistic dựng lại đúng log-odds, chọn trial tốt nhất, fingerprint ổn định |

Toàn bộ: 409 test pass.

## 7. Giới hạn

- Split theo thời gian, không theo khoản vay: cùng một khoản có thể có mặt ở nhiều đoạn (ở các tháng khác nhau).
- PD không hiệu chỉnh lại; mức PD lệch theo đoạn (mục 4.4).
- Cờ deferral gần như chỉ có từ 2020; forbearance và modification hiếm trong train. Mô hình học rất ít từ chúng,
  và đó là một lý do capture giảm mạnh ở stress test.
- Không có biến vĩ mô (lãi suất thị trường, giá nhà, thất nghiệp).
- Survival chỉ mô tả, không covariate; giả định censoring không mang thông tin; khoản `other_exit` coi là censored.
- Dữ liệu là Freddie Mac sample (50,000 khoản mỗi lứa), không phải toàn bộ danh mục.

## 8. Tái lập

```bash
python -m credit_risk run --steps ews --source freddie_mac
```

Cần đã chạy `ingest` cho Freddie Mac. Bước `ews` tự dựng lại monitoring panel nên không phụ thuộc lần chạy
`monitor`. Cần khoảng 3-4 GB RAM trống; giới hạn RAM của DuckDB đặt trong `.env`.
