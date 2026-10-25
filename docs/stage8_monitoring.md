# Stage 8: Portfolio monitoring (Freddie Mac)

Chặng này trả lời câu hỏi đầu của Module 3: **danh mục xấu đi ở đâu?** Một lệnh `monitor` dựng panel giám sát
từ 35.6 triệu dòng tháng, tính chỉ số danh mục hằng tháng, vintage curve, roll rate và migration, phân khúc và
mức tập trung, rồi áp các quy tắc cảnh báo trong `ews.yaml` để ra trạng thái xanh / vàng / đỏ.

Con số dưới đây lấy từ lần chạy `20261025T093046Z-27459ab4`, tháng báo cáo **2026-03**: **38 kiểm tra, 0 lệch;
trạng thái đỏ (2 đỏ, 7 vàng, 60 xanh)**. Toàn bộ bước chạy khoảng 40 giây.

> **Cách đọc nhanh:** chỉ số tháng so với trung bình 12 tháng trước cho biết danh mục có *đột ngột* xấu đi
> không. Vintage curve so các lứa ở **cùng MOB**, cho biết khoản giải ngân gần đây có *tệ hơn* khoản cũ không.
> Hai góc nhìn này trả lời hai câu hỏi khác nhau, và lần này chúng cho hai kết quả khác nhau.

## 1. Kết luận nhanh

- **Tháng 2026-03 không có biến động đột ngột.** Cả mười chỉ số danh mục và roll rate đều xanh. Tỷ lệ 90+ là
  0.628% theo số khoản và 0.688% theo dư nợ, so với 0.583% / 0.625% trung bình 12 tháng trước (+8% / +10%).
  Roll rate thấp hơn mốc, cure rate 30 → current cao hơn (45.4% so với 38.5%).
- **Nhưng lứa mới tệ hơn rõ rệt (đỏ).** Ở MOB 24, default cộng dồn của lứa 2023 là 1.77% so với 0.56% của các
  lứa 2012-2016. Ở MOB 12, lứa 2024 là 0.60% so với 0.17%. Hai lứa 2022 và 2023 cũng ở mức khoảng 1.8% tại MOB 24.
- **Bảy cảnh báo vàng theo phân khúc:** nhóm LTV 96-100 (90+ là 1.71% so với 1.39%) và sáu bang (IA, LA, MD,
  NC, PA, TN), với tỷ lệ 90+ tăng 23-44% so với 12 tháng trước.
- **Mặt bằng sau COVID cao hơn trước COVID:** 90+ trung bình 0.58% so với 0.19%, default mới mỗi tháng 0.047%
  so với 0.026%.
- **Danh mục không tập trung bất thường:** bang lớn nhất là CA với 13.5% dư nợ (HHI theo bang 0.048). PSI cơ cấu
  so với 12 tháng trước xanh ở mọi chiều, trừ `vintage`, chiều này không được xét (mục 4.5).

## 2. Quyết định (2026-10-25)

| # | Vấn đề | Quyết định | Ghi ở |
|---|---|---|---|
| 1 | Tỷ lệ theo số khoản hay theo dư nợ | **Cả hai**, cho mọi chỉ số | `ews.yaml -> monitoring.weights` |
| 2 | Khoản bị cắt trái trong vintage curve | **Loại khỏi curve** (dòng đầu tiên có `loan_age` > 1), có đếm số khoản bị loại; xử lý delayed entry để dành cho chặng 9 | `vintage_curve.max_first_loan_age` |
| 3 | Band FICO / LTV / DTI | **Band thị trường**: FICO <620 / 620-679 / 680-739 / 740-779 / 780+; LTV ≤60 / 61-80 / 81-90 / 91-95 / 96-100 / >100; DTI ≤36 / 37-43 / 44-50 / >50; giá trị thiếu là một band riêng | `monitoring.bands` |
| 4 | Mốc so sánh cho cảnh báo | **Trung bình 12 tháng ngay trước** tháng báo cáo; vàng khi xấu hơn mốc trên 20%, đỏ khi trên 50% | `monitoring.alerts` |

Quy tắc 2 loại 20,416 khoản (2.9%), không phải 8.7% như con số "khoản bắt đầu ở MOB > 0" trong tài liệu chất
lượng dữ liệu: 40,253 khoản bắt đầu ở MOB 1, mà đó là quy ước báo cáo chứ không phải lịch sử bị mất.

## 3. Thứ tự và kiểm tra

| # | Module | Câu hỏi | Kiểm tra | Dừng nếu lệch? |
|---|---|---|---|---|
| 8.0 | `panel` | Panel có đúng là dữ liệu nguồn? | 10 | Có |
| 8.1 | `portfolio` | Danh mục tháng này ra sao? | 7 | Không |
| 8.2 | `vintage` | Lứa mới so với lứa cũ ở cùng tuổi? | 4 | Không |
| 8.3 | `roll` | Khoản chuyển bucket thế nào? | 4 | Không |
| 8.4 | `concentration` | Rủi ro dồn vào đâu? | 13 | Không |
| 8.5 | `alerts` | Có gì cần hành động? | | |

Mọi module đọc `monitoring_panel.parquet`, một dòng cho mỗi cặp khoản vay - tháng. Bảng này được dựng bằng
DuckDB (35.6 triệu dòng mất 22 giây) và các tổng hợp cũng chạy trong DuckDB. Những kiểm tra quan trọng nhất đối
chiếu với **một đường tính khác**, không chỉ với chính panel:

- bucket tính bằng SQL so với `status_to_bucket` (bản pandas mà test nhãn EWS cố định) trên cả 91 mã trạng thái
  có trong dữ liệu;
- tổng số khoản trả hết nợ / thanh lý cộng qua các tháng so với số đếm `zero_balance_code` trên file gốc. Kiểm
  tra này đã **bắt được một lỗi thật** ở lần chạy đầu: `has_prev` bị NULL ở dòng đầu của mỗi khoản, làm rơi mất
  547 khoản trả hết nợ;
- số khoản 90+ của tháng báo cáo so với số đếm từ mã trạng thái gốc;
- default ở MOB 12 theo từng lứa so với một lần đếm lại bằng SQL;
- migration của tháng báo cáo so với `transitions.transition_matrix` (pandas, ghép tháng bằng merge thay vì
  window function) trên mẫu 20,000 khoản;
- theo từng chiều phân khúc, tổng các nhóm bằng tổng danh mục.

## 4. Kết quả

### 4.1 Danh mục tháng (8.1)

| Chỉ số | 2026-03 | TB 12 tháng trước |
|---|---:|---:|
| Khoản đang hoạt động | 325,219 | 324,722 |
| Dư nợ đang hoạt động | 78.6 tỷ USD | 78.5 tỷ USD |
| 30+ (số khoản / dư nợ) | 1.803% / 1.813% | 1.793% / 1.777% |
| 90+ (số khoản / dư nợ) | 0.628% / 0.688% | 0.583% / 0.625% |
| Default mới mỗi tháng (số khoản) | 0.051% | 0.054% |
| CPR (số khoản / dư nợ) | 12.9% / 15.3% | 9.1% / 9.4% |

| Giai đoạn | Số tháng | 30+ | 90+ | Default mới mỗi tháng | CPR |
|---|---:|---:|---:|---:|---:|
| Trước COVID | 98 | 0.73% | 0.19% | 0.026% | 8.2% |
| COVID (2020-03 → 2021-12) | 22 | 3.44% | 2.06% | 0.224% | 27.6% |
| Sau COVID | 51 | 1.65% | 0.58% | 0.047% | 8.0% |

Cách tính:
- **Đầu kỳ:** các khoản còn hoạt động ở cuối tháng trước.
- **Tỷ lệ quá hạn:** tính trên các khoản còn hoạt động có trạng thái đã biết.
- **Default mới:** lần đầu tiên khoản vay vào 90+ hoặc bị thanh lý; tái vỡ nợ sau khi cure không tính là mới.
- **SMM / CPR:** chỉ tính khoản trả hết nợ.

### 4.2 Vintage curve (8.2)

Default cộng dồn (số khoản) tại các mốc MOB:

| Lứa | MOB 12 | MOB 24 | MOB 36 | MOB 60 |
|---|---:|---:|---:|---:|
| 2012-2016 | 0.14-0.22% | 0.49-0.68% | 0.79-1.00% | 1.36-4.30% |
| 2017 | 0.40% | 0.82% | **3.38%** | 5.38% |
| 2018 | 0.22% | **2.60%** | 4.70% | 5.17% |
| 2019 | **2.33%** | 4.59% | 4.93% | 5.27% |
| 2020 | 1.25% | 1.62% | 1.83% | 2.16% |
| 2021 | 0.37% | 0.91% | 1.37% | - |
| 2022 | 0.70% | 1.84% | 2.98% | - |
| 2023 | 0.62% | 1.77% | - | - |
| 2024 | 0.60% | - | - | - |

COVID hiện thành một **đường chéo**: lứa 2017 nhảy ở MOB 36, lứa 2018 ở MOB 24, lứa 2019 ở MOB 12, cùng rơi vào
năm 2020. Lứa 2022-2024 không chịu COVID nhưng vẫn cao gấp khoảng 3 lần lứa 2012-2016 ở cùng MOB.

Một quy tắc đã được sửa sau khi chạy thật. Lúc đầu, mỗi curve dừng ở MOB mà *mọi* khoản trong pool đã đạt tới,
và quy tắc này cắt cụt curve của lứa 2022-2024 tới hai năm. Nguyên nhân: dưới 0.1% khoản mỗi lứa có `loan_age = 0`
muộn hơn quý ghi trong `loan_id` cả năm trở lên. Curve giờ dừng ở MOB mà 99% pool đã đạt tới
(`min_reach_share`), nên đoạn đuôi chỉ có thể bị kéo thấp tối đa 1% (tương đối).

### 4.3 Roll rate và migration (8.3)

| Chuyển (số khoản) | 2026-03 | TB 12 tháng trước | Trước COVID | COVID | Sau COVID |
|---|---:|---:|---:|---:|---:|
| current → 30 | 0.44% | 0.52% | 0.33% | 0.70% | 0.51% |
| 30 → 60 | 14.4% | 17.1% | 13.1% | 35.6% | 16.5% |
| 60 → 90+ | 32.1% | 36.7% | 34.5% | 62.0% | 38.2% |
| 30 → current | 45.4% | 38.5% | 49.1% | 35.9% | 42.7% |

Ba cột theo giai đoạn là ma trận gộp của từng giai đoạn. Thời COVID, một khoản 60 ngày có gần hai phần ba khả năng
sang 90+. Mức sau COVID cao hơn trước COVID ở mọi bước chuyển xấu đi.

### 4.4 Phân khúc và tập trung (8.4)

Tỷ lệ 90+ tháng 2026-03 theo FICO: 2.19% ở 620-679, 1.04% ở 680-739, 0.36% ở 740-779 và 0.15% ở 780+. Theo LTV,
tỷ lệ tăng đều từ 0.36% (≤60) lên 1.71% (96-100). Theo DTI: 0.40% (≤36) so với 0.95% (44-50). Kênh môi giới (B)
cao nhất với 0.72%. Bang lớn nhất là CA (13.5% dư nợ); FL (0.97%) và TX (0.80%) là hai bang lớn có tỷ lệ 90+ cao.

### 4.5 Cảnh báo (8.5)

| Loại | Nhìn vào | Mốc |
|---|---|---|
| `metric` | 10 chỉ số danh mục và roll rate của tháng | trung bình 12 tháng trước; cure rate: giảm mới là xấu |
| `segment` | 90+ của từng nhóm phân khúc có từ 2,000 khoản trở lên | tỷ lệ gộp 12 tháng trước của chính nhóm |
| `mix_psi` | cơ cấu dư nợ theo từng chiều | cùng tháng năm trước; ngưỡng PSI trong `definitions.yaml` |
| `new_vintage` | default cộng dồn MOB 12 và 24 của lứa gần nhất đã đạt tới | trung bình các lứa 2012-2016 |

PSI cơ cấu theo **lứa** là 0.33 (đỏ) nhưng được ghi "không xét". Mẫu Freddie có 50,000 khoản mới mỗi năm còn các
lứa cũ dần tất toán, nên cơ cấu theo lứa luôn trôi do cách thiết kế mẫu, không phải do chính sách cho vay.

Mốc của quy tắc `new_vintage` là các lứa 2012-2016, giai đoạn lãi suất thấp và trước COVID. Vì vậy cảnh báo đỏ ở
đây nói rằng *khoản mới tệ hơn chuẩn lịch sử*. Nó không tách được bao nhiêu là do khẩu vị cho vay, bao nhiêu là do
môi trường lãi suất 6-7%. Muốn đổi chuẩn so sánh thì sửa `benchmark_vintages`.

## 5. Đầu ra

`artifacts/<run_id>/monitor/`:

| File | Nội dung |
|---|---|
| `monitoring_metrics.json` | `provenance` (run, config hash, tháng báo cáo, seed, tóm tắt panel, số kiểm tra, trạng thái, số cảnh báo theo mức), `status`, `alerts`, `checks`, `limitations`, `sections` (danh mục mọi tháng, roll rate, vintage theo mốc MOB, migration của tháng và theo giai đoạn, phân khúc, tập trung) |
| `monitoring_report.md` | Trạng thái và cảnh báo, phạm vi và giới hạn, kiểm tra, rồi từng module |
| `portfolio_monthly.csv`, `vintage_curves*.csv`, `transition_counts.csv`, `roll_rates.csv`, `segments_monthly.csv`, `concentration.csv`, `alerts.csv`, `checks.csv` | Bảng đầy đủ cho dashboard (chặng 10) |

`data/processed/freddie_mac/monitoring_panel.parquet` là đầu vào dùng chung cho EWS (chặng 9).

## 6. Definition of Done: bằng chứng

`tests/test_stage8_dod.py` chạy bước thật qua pipeline trên một panel tổng hợp: 3 lứa, 1,200 khoản chuyển bucket
theo xích Markov, có trả trước, roll lên 90+, thanh lý, và cứ 20 khoản thì một khoản bị cắt trái. Các test:

- một lệnh cho đủ JSON, báo cáo và bảng CSV, manifest ghi đủ đầu ra; mọi kiểm tra đạt; provenance khớp config hash
  và manifest; đếm đúng 60 khoản bị cắt trái;
- chạy hai lần cho cùng metrics (trừ run id);
- báo cáo đúng thứ tự Trạng thái → Cảnh báo → Mọi quy tắc → Phạm vi và giới hạn → kiểm tra → các module; mọi
  cảnh báo vàng / đỏ có trong bảng cảnh báo;
- **ngưỡng quyết định mức cảnh báo:** ngưỡng rất lỏng cho trạng thái xanh, ngưỡng rất chặt cho đỏ;
- `reporting_month` đặt trong config được dùng đúng;
- nhân đôi một dòng tháng trong `performance.parquet` làm bước thất bại với lý do "no duplicated (loan, month)".

Test module (36 test): `test_monitoring_panel.py` (7), `test_monitoring_portfolio.py` (6),
`test_monitoring_vintage.py` (4), `test_monitoring_roll.py` (6), `test_monitoring_concentration.py` (5),
`test_monitoring_alerts.py` (8). Các con số được tính tay, mỗi quy tắc được kiểm ở hai phía ngưỡng, và mỗi kiểm
tra đều có kịch bản cố tình làm hỏng để xác nhận nó bắt được lỗi.

## 7. Giới hạn

- Dữ liệu là mẫu 50,000 khoản mỗi lứa, không phải toàn bộ danh mục; cơ cấu theo lứa đổi theo cách thiết kế mẫu.
- SMM / CPR chỉ tính khoản trả hết nợ (dữ liệu không có trả trước một phần).
- Cảnh báo so với 12 tháng trước là tín hiệu tương đối: một danh mục xấu đi chậm và đều sẽ không bị cảnh báo.
  Vintage curve và bảng theo giai đoạn bù cho điểm này.
- Chặng này mô tả danh mục, không dự báo. Dự báo khoản nào sắp vào 90+ là việc của EWS (chặng 9).

## 8. Tái lập

```
python -m credit_risk run --steps monitor --source freddie_mac
```

Cần `data/processed/freddie_mac/{origination,performance}.parquet` (bước `ingest`). Khoảng 40 giây trên laptop
7.4 GB RAM.
