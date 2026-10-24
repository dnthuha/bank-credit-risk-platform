# Stage 6: Validation engine (Home Credit)

Chặng này đổi vai: không còn là người xây mô hình mà là bên thẩm định độc lập. Gói model đóng băng ở Stage 5
chỉ là **đầu vào**, không phải sự thật. Mọi con số được tính lại từ dữ liệu đã xử lý và từ chính gói, bằng
code riêng trong `credit_risk.validation` (numpy / pandas / DuckDB, không gọi sklearn ở phần đo lường).
Con số dưới đây lấy từ lần chạy chính thức `20261024T171209Z-eb995677` trên gói `c95dc90913d8`. Sau khi sửa
phát hiện 2 (chặng 7.0, 2026-10-25) và chạy lại `calibrate` + `validate-model` (run `20261024T173307Z-eb995677`,
fingerprint gói không đổi): **252 kiểm tra, 0 lệch, 17 phát hiện**.

> **Cách đọc nhanh:** "kiểm tra" là một phép so sánh có đáp án (công bố so với tính lại, hoặc một bất biến như
> Gini = 2·AUC − 1). "Phát hiện" là một điều người phê duyệt cần biết, có hay không kèm kiểm tra lệch. Kết
> luận Đạt / Đạt có lưu ý / Phải khắc phục là việc của chặng 7; chặng này chỉ đo và liệt kê.

## 1. Kết luận nhanh

- **Một lệnh, hai file:** `python -m credit_risk run --steps validate-model --source home_credit` chạy 8 module
  trong khoảng 30 giây, ghi `artifacts/<run_id>/validate-model/metrics.json` (máy đọc) và
  `validation_report.md` (người đọc). Chạy lại cho cùng metrics.
- **252 kiểm tra, 2 lệch, 19 phát hiện.** Hai lệch đều là slope hiệu chỉnh lần phát triển báo ở tolerance mặc
  định của solver (lệch 0.0011 và 0.0002), không ảnh hưởng mô hình.
- **Gói đúng là gói công bố:** SHA-256 từng file, fingerprint ba cấp tính lại, 5,000 hồ sơ chấm lại trùng
  từng bit về score, PD và 8 mã lý do.
- **Dân số khớp tuyệt đối** (41 kiểm tra, dung sai 0 dòng): 307,511 hồ sơ, không loại trừ, chia
  184,507 / 30,750 / 46,127 / 46,127.
- **Trên tập test** (chưa ai đọc trước chặng này): champion Gini 0.505 [0.488, 0.520], challenger 0.563
  [0.547, 0.578]. Challenger tốt hơn có ý nghĩa (+0.058, khoảng [+0.049, +0.067]). Mức tụt từ validation sang
  test (0.011 / 0.017) nằm trong nhiễu mẫu.
- **Ba điểm cần người phê duyệt cân nhắc:** (a) champion dự báo thiếu rủi ro ở band PD thấp nhất và theo tuổi
  lệch rõ cả hai chiều; (b) biến `APP_ANNUITY_TO_CREDIT` của challenger có PSI 1.00 trên application_test;
  (c) người dưới 30 bị từ chối gấp 5 lần người trên 60, AIR 0.56 / 0.66, một phần vượt ngoài bad rate thật.

## 2. Nguyên tắc và thứ tự

| # | Module | Câu hỏi | Dừng nếu lệch? |
|---|---|---|---|
| 6.1 | `package` | Đây có đúng là gói đã công bố, chưa bị sửa? | Có |
| 6.2 | `population` | Mẫu có đúng là mẫu model card mô tả, không loại trừ ngầm? | Có |
| 6.3 | `discrimination` | AUC / Gini / KS tính lại có bằng công bố? KS xảy ra ở đâu? | Không |
| 6.4 | `calibration_check` | PD đúng mức ở đâu, lệch hệ thống ở đâu? | Không |
| 6.5 | `stability` | Dân số mới có khác dân số học? Biến nào trôi? | Không |
| 6.6 | `uncertainty` | Các con số trên chắc đến đâu? | Không |
| 6.7 | `segments` | Có nhóm nào bị con số tổng che đi? Rủi ro công bằng? | Không |
| 6.8 | `implementation` | Gói chạy lại có cho đúng đầu ra đã công bố? | Không |

Hai bước đầu dừng pipeline khi lệch vì mọi con số sau đó sẽ nói về một mô hình hoặc một mẫu khác với mô hình
và mẫu trong model card. Các bước sau ghi lệch thành phát hiện và chạy tiếp.

Mỗi module đọc **gói** (`model_package/`) và **dữ liệu đã xử lý** (`features.parquet`, `splits.parquet`,
`application_*.parquet`), không đọc báo cáo của Module 1. Các file của lần chạy phát triển
(`artifacts/<run>/train/*.json`, `calibrate/calibration.json`, `features/splits_summary.json`) chỉ dùng để
**đối chiếu** con số đã công bố, khi còn trên máy; không có thì phần đối chiếu bị bỏ qua và ghi rõ.

## 3. Kết quả từng module

### 6.1 Gói model: 15/15

SHA-256 bốn file khớp `package.json`; scorecard và LightGBM dựng lại từ file rồi tính lại fingerprint, khớp
file lẫn `package.json`; thang điểm trong gói khớp `definitions.yaml`; challenger ghi đúng champion nó được
train cạnh. Sửa một byte hay sửa một hệ số rồi vá SHA đều bị bắt (test).

**Phát hiện 1 (đã sửa ở 6.0).** Chạy lại Stage 3-5 trên máy khác cho đúng mọi hệ số, điểm, Gini, calibrator
nhưng fingerprint khác, vì hash băm hệ số lbfgs ở độ chính xác đầy đủ (lệch chữ số 15 giữa các máy).
Fingerprint nay băm số thực làm tròn 8 chữ số (`lineage.round_floats`). Docs Stage 3-5 đã ghi chú.

### 6.2 Dân số: 41/41, dung sai 0 dòng

Đếm lại bằng DuckDB và so với: số hồ sơ train / validation hai mô hình tự khai, split fingerprint trong cả ba
file của gói, số hồ sơ đã chấm theo từng split và application_test, tỷ lệ so với 60 / 10 / 15 / 15, bad rate từng
split so tổng thể (stratified), 9 bộ đếm loại trừ phải bằng 0, và `splits_summary.json` của lần chạy features.

### 6.3 Sức phân biệt: 32/32

| Mô hình | Tập | AUC | Gini | KS | KS tại | Cặp hòa |
|---|---|---:|---:|---:|---|---:|
| champion | validation | 0.7576 | 0.5153 | 0.3882 | score ≥ 559 | 0.7% |
| champion | **test** | 0.7523 | 0.5046 | 0.3904 | score ≥ 556 | 0.7% |
| challenger | validation | 0.7898 | 0.5795 | 0.4413 | PD ≤ 0.0704 | 0.0% |
| challenger | **test** | 0.7813 | 0.5625 | 0.4306 | PD ≤ 0.0800 | 0.0% |

AUC tính hai cách (theo hạng và đếm cặp) trùng tới 1e-12; Gini = 2·AUC − 1 đúng trên mọi dòng tính lại và mọi
cặp công bố; 16 con số công bố (train, validation) khớp trong 1e-6.

### 6.4 Hiệu chỉnh: 74 kiểm tra, 2 lệch

Slope và intercept giải bằng Newton tới 1e-12, có sai số chuẩn; intercept-in-the-large; 10 band theo PD với
z-score; Hosmer-Lemeshow.

| Mô hình | Tập | Bad rate | PD TB | Brier | Slope (SE) | Intercept lớn | HL p | Band lệch |
|---|---|---:|---:|---:|---|---:|---:|---|
| champion | **test** | 8.07% | 8.08% | 0.06845 | 0.964 (0.019) | -0.077 | 0.005 | 1 thiếu (z +3.6), 6 thừa (z -2.2) |
| challenger | **test** | 8.07% | 8.08% | 0.06640 | 0.971 (0.017) | -0.059 | 0.055 | 8 thiếu (z +2.7) |

**Phát hiện 2.** Slope lần phát triển (`calibration.json`) lệch 0.0011 (champion) và 0.0002 (challenger) so với
nghiệm hợp lý cực đại: sklearn với tolerance mặc định dừng sớm; chạy lại sklearn với `tol=1e-10` cho đúng số
của Newton. **Đã sửa ở chặng 7.0:** `models/calibration.calibration_slope_intercept` dùng `tol=1e-10`; hai kiểm tra
slope nay khớp (chỉ đổi con số báo cáo, không đổi gói). Mean PD, bad rate, Brier, intercept-in-the-large và 40 con số bảng reliability khớp.

**Phát hiện 3.** Band PD thấp nhất của champion trên test: kỳ vọng 50.5 bad, quan sát 76 (bad rate 1.65% so
PD 1.09%). Nhóm "an toàn nhất" theo scorecard thực tế rủi ro hơn 50% so với PD nói. Slope 0.964 thấp hơn 1
gần 2 SE, nhất quán với PD hơi cực đoan ở hai đầu. Tổng thể vẫn đúng mức (tỷ số 0.999).

### 6.5 Độ ổn định: 42/42, tham chiếu train

| Hạng mục | So với application_test | Đèn |
|---|---:|---|
| Score champion, tổng thể | 0.0013 | xanh |
| PD challenger, tổng thể | 0.0026 | xanh |
| Score champion, Revolving loans (439 hồ sơ) | 0.1602 | vàng |
| PD challenger, Revolving loans | 0.1151 | vàng |
| `APP_ANNUITY_TO_CREDIT` (biến #5 challenger theo gain) | **1.0015** | **đỏ** |
| `EXT_SOURCE_1` (#3 cả hai; thiếu 56% → 42%) | 0.0824 | xanh |

Ba tập cùng kỳ (test, validation, calibration) có PSI ≤ 0.004 ở mọi hạng mục: split tốt, phương pháp đúng. Sáu
con số PSI công bố khớp.

**Phát hiện 4.** Hồ sơ application_test có tỷ lệ khoản trả / số vay trung vị 0.061 so 0.050 ở train (kỳ hạn
ngắn hơn); bin thấp nhất của train (9.7%) gần trống, hai bin cao gấp đôi. Challenger đọc biến này ở vùng ít được
học. Scorecard không dùng biến này. Đề xuất: theo dõi trong monitoring (chặng 8); cân nhắc loại hoặc cắt bin cho
challenger nếu triển khai.

### 6.6 Độ bất định: 24/24, bootstrap phân tầng 1000 lần, seed 2026

| So sánh | Chỉ số | Chênh | Khoảng 95% | Khác 0? |
|---|---|---:|---|---|
| challenger − champion, test (ghép đôi) | Gini | +0.058 | [+0.049, +0.067] | có |
| challenger − champion, test (ghép đôi) | KS | +0.040 | [+0.029, +0.053] | có |
| validation − test, champion | Gini | +0.011 | [-0.015, +0.035] | không |
| validation − test, challenger | Gini | +0.017 | [-0.007, +0.039] | không |

Cả hai mô hình được đánh giá trên cùng những hàng tái chọn, nên khoảng của chênh lệch hẹp hơn hai khoảng riêng
cộng lại. Ước lượng điểm trùng 6.3 tới 1e-12; chạy lại cho cùng khoảng từng bit.

### 6.7 Phân khúc và rà soát công bằng: 13/13 kiểm tra, 12 phát hiện

Quyết định ghi trong `validation.yaml -> segment_options` (2026-10-24): thẩm định cả hai mô hình; dải tuổi
< 30 / 30-45 / 45-60 / ≥ 60; "từ chối" = PD đã calibrate > 10% (cùng quy tắc cho hai mô hình); nhóm dưới 500
hồ sơ không tính AUC; AIR = tỷ lệ được duyệt của nhóm / nhóm được duyệt nhiều nhất (quy tắc 4/5).

- **Giới tính:** hiệu chỉnh đúng trong nhiễu ở cả hai mô hình (z trong ±0.6). Nam bị từ chối 36.9% so nữ 21.3%
  (champion), AIR 0.80; bad rate thật 10.2% so 7.0% và PD khớp bad rate ở cả hai nhóm.
- **Tình trạng hôn nhân:** hiệu chỉnh trong nhiễu. Civil marriage và Single AIR 0.76 (champion), 0.80 / 0.83
  (challenger). Widow: AUC thấp hơn tổng thể 0.044 / 0.035 (2,431 hồ sơ).
- **Tuổi là điểm yếu (Phát hiện 5):**

| Nhóm | Hồ sơ | Bad rate | Champion bad/PD, z | Champion AUC | Từ chối | AIR |
|---|---:|---:|---|---:|---:|---:|
| < 30 | 6,816 | 11.75% | 0.93, z -2.3 (PD cao) | 0.716 | 49.2% | 0.56 |
| 30-45 | 18,697 | 9.16% | 1.09, z +3.7 (PD thấp) | 0.756 | 28.4% | 0.79 |
| 45-60 | 15,388 | 6.10% | 0.90, z -3.5 (PD cao) | 0.755 | 20.2% | 0.88 |
| ≥ 60 | 5,226 | 5.19% | 1.11, z +1.8 | 0.668 | 9.8% | 1.00 |

Champion đánh giá nhóm 30-45 quá an toàn, hai nhóm 45-60 và < 30 quá rủi ro, vượt nhiễu rõ. Challenger chỉ lệch
ở 45-60 (z -3.2), AUC < 30 là 0.748 và ≥ 60 là 0.724. Người dưới 30 bị từ chối gấp 5 lần người trên 60; một
phần do bad rate thật cao hơn gấp đôi, nhưng champion còn dự báo thừa rủi ro cho nhóm này nên mức từ chối có phần
không được bad rate giải thích. Đây là hệ quả trực tiếp của quyết định PROJECT_SCOPE #7 (giữ `DAYS_BIRTH`,
`CODE_GENDER`, `NAME_FAMILY_STATUS` trong scorecard) và là điểm quan trọng nhất cho kết luận chặng 7.

### 6.8 Đối chiếu triển khai: 11/11

5,000 hồ sơ (seed 2026, phân bổ theo population và split: train 2,590, application_test 684, calibration 647,
test 647, validation 432) chấm lại từ gói: score trùng tuyệt đối, hai PD lệch tối đa 0, 8 mã lý do trùng đúng
thứ tự. Không chênh lệch.

## 4. Đầu ra

`artifacts/<run_id>/validate-model/`:

| File | Nội dung |
|---|---|
| `metrics.json` | `provenance` (fingerprint gói / champion / challenger / split, run của gói, config hash, seed, số lần bootstrap, cỡ mẫu đối chiếu, số hồ sơ từng tập, số kiểm tra và lệch), `checks` (252 dòng: module, kiểm tra, công bố, tính lại, nguồn, ok), `findings` (module, mức độ, nội dung), `sections` (mọi bảng của 8 module dạng records) |
| `validation_report.md` | Cùng nội dung cho người đọc: tiêu đề có fingerprint và seed, bảng kiểm tra theo module, danh sách phát hiện, kiểm tra lệch, rồi từng module |

Mức độ phát hiện: `blocking` (gói hoặc dân số không khớp, chấm lại lệch), `major` (PSI đỏ), `fairness` (nhóm
nhạy cảm bị cờ hiệu chỉnh, AIR < 0.8 hoặc AUC lệch > 0.02), `note` (còn lại). Quy tắc biến chúng thành kết
luận nằm ở `validation.yaml -> conclusion_rules`, chặng 7.

## 5. Definition of Done: bằng chứng

`tests/test_stage6_dod.py` chạy bước thật trên gói tổng hợp (fixture `stage6_build`, 20,000 hồ sơ):

- một lệnh cho `metrics.json` và `validation_report.md` phủ đủ 8 module, manifest ghi hai file;
- `provenance` khớp fingerprint trong `package.json`, config hash trong manifest, seed và số lần bootstrap trong
  `validation.yaml`;
- chạy hai lần cho cùng metrics (trừ run id);
- sửa một byte `challenger_model.txt`: bước thất bại, manifest nêu đúng kiểm tra SHA;
- xóa một hồ sơ khỏi `application_train`: bước thất bại với "population does not reconcile";
- mọi phát hiện có module và mức độ hợp lệ; số kiểm tra lệch và số phát hiện trong `provenance` khớp danh sách.

Test module (8 file `tests/test_validation_*.py`, 62 test): ví dụ tính tay cho từng chỉ số, đối chiếu sklearn /
scipy trên dữ liệu ngẫu nhiên nhiều hòa, mô hình hoàn hảo và ngẫu nhiên, Newton khôi phục méo đã biết, độ phủ
bootstrap, và kịch bản bị sửa (file, hệ số vá SHA, hồ sơ thiếu / trùng / chuyển split, con số công bố bị sửa,
biến bị dịch, nhóm bị dự báo thiếu, điểm công bố bị sửa) đều bị bắt đúng chỗ.

## 6. Giới hạn

- Không có out-of-time: test là mẫu ngẫu nhiên cùng kỳ. Độ ổn định theo thời gian chỉ đo được trên Freddie Mac
  (chặng 8-9).
- application_test không có TARGET nên chỉ đo PSI, không đo hiệu năng trên dân số mới.
- Khoảng tin cậy bootstrap là percentile đơn giản; chưa hiệu chỉnh đa so sánh cho 12 phát hiện công bằng.
- Chặng này không kết luận. Ngưỡng "đạt" cho slope, PSI, AIR, AUC theo nhóm phải được chốt ở chặng 7.

## 7. Tái lập

```
python -m credit_risk run --steps validate-model --source home_credit
```

Khoảng 30 giây khi đã có gói Stage 5 và dữ liệu đã xử lý. Đối chiếu với con số công bố chỉ đầy đủ khi
`artifacts/<run của gói>/` còn trên máy.
