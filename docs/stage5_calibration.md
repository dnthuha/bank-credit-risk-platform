# Stage 5: Calibration và đóng băng model (Home Credit)

Chặng này kiểm tra và chỉnh để PD của hai mô hình **đúng về mức**, rồi gói cả hai thành một model package
cố định cho chặng 6 thẩm định. Mọi con số dưới đây lấy từ lần chạy chính thức `20260927T103126Z-e9c612e9`
(commit `ce14115`, working tree sạch), trên champion `a254458deddf` và challenger `78b3844ac54f` của Stage 3-4.

> **Cách đọc nhanh:** AUC / Gini đo thứ hạng; calibration đo con số. Một mô hình có thể xếp hạng đúng hoàn
> toàn mà PD vẫn sai (ví dụ nói 10% khi thực tế là 15%). PD đúng mức là điều kiện để đặt ngưỡng duyệt, định
> giá theo rủi ro và tính tổn thất dự kiến.

## 1. Kết luận nhanh

- **Cả hai mô hình vốn đã gần như được calibrate.** Trên validation trước khi chỉnh: champion slope 0.971,
  PD trung bình 8.28% so với bad rate 8.07%; challenger slope 1.000, 8.01% so với 8.07%.
- **Cả hai chọn Platt.** Sau calibration, trên validation: champion slope 1.001, intercept +0.014;
  challenger slope 1.013, intercept +0.011. AUC không đổi (0.7576 và 0.7898).
- Isotonic không đáng: với champion chỉ tốt hơn Platt 0.00004 Brier (nhiễu) nhưng làm AUC giảm 0.002; với
  challenger còn kém hơn cả không calibrate.
- Challenger **trên train** có slope 1.339 (quá tự tin vì overfit): đúng lý do calibration phải làm trên tập riêng.
- Model package đóng băng, fingerprint `35e42a5f208b`; năm lần chạy (bốn lần trong lúc phát triển, một lần
  từ commit sạch `ce14115`) cho cùng fingerprint và cùng SHA-256 từng file.

## 2. Dữ liệu

| Tập | Dùng để | Ghi chú |
|---|---|---|
| calibration (15%, 46,127 hồ sơ, 3,720 bad) | Fit calibrator; chọn phương pháp bằng 5-fold CV bên trong | Chưa mô hình nào đọc tới trước chặng này |
| validation (10%) | Kiểm tra độc lập sau calibration | Không dùng để fit calibrator |
| train | Chỉ để báo cáo mức lệch trước calibration | |
| test (15%) | **Không đọc** | Dành cho chặng 6 |

## 3. Phương pháp

Đầu vào của calibrator là log-odds gốc của mô hình (scorecard: suy từ điểm qua thang điểm; LightGBM:
logit của PD thô).

| Cách | Công thức | Ưu | Nhược |
|---|---|---|---|
| Không calibrate | PD gốc | Mốc so sánh | Không bao giờ được chọn |
| Platt | PD = sigmoid(a + b x log-odds) | 2 tham số, khó overfit, giữ nguyên thứ hạng | Chỉ sửa lệch dạng chữ S |
| Isotonic | Hàm bậc thang không giảm | Sửa mọi dạng lệch | Dễ overfit, gộp hồ sơ vào cùng mức PD nên AUC giảm |

**Quy tắc chọn** (`model_dev.yaml` -> `calibration`): 5-fold CV trong tập calibration, chấm Brier ngoài fold.
Các phương pháp cách phương pháp tốt nhất không quá 0.0001 Brier coi như bằng nhau; chọn cái đơn giản hơn.
Phương pháp thắng được fit lại trên toàn bộ tập calibration.

"Không calibrate" không được phép thắng: Brier kém nhạy với lệch về mức trung bình. Với champion, không
calibrate chỉ kém Platt 0.00003 Brier, trong khi PD trung bình đang cao hơn bad rate khoảng 4%.

## 4. Kết quả

| | Champion | Challenger |
|---|---|---|
| Brier ngoài fold: không / Platt / isotonic | 0.06804 / 0.06801 / 0.06797 | 0.06599 / 0.06598 / 0.06604 |
| Chọn | Platt: a = -0.1055, b = 0.9702 | Platt: a = -0.0270, b = 0.9873 |
| Validation, PD TB trước -> sau (bad rate 8.07%) | 8.28% -> 7.97% | 8.01% -> 8.00% |
| Validation, slope trước -> sau | 0.971 -> 1.001 | 1.000 -> 1.013 |
| Validation, Brier trước -> sau | 0.06808 -> 0.06806 | 0.06570 -> 0.06569 |

Reliability trên validation (10 nhóm theo PD đã calibrate): bad rate / PD của champion nằm trong 0.93 - 1.14
ở mọi nhóm, 1.00 ở nhóm rủi ro nhất. Challenger lệch nhiều hơn ở hai nhóm PD thấp nhất (0.79 và 0.81: PD
hơi cao so với thực tế), khoảng 25 bad mỗi nhóm nên còn trong dao động mẫu; isotonic không sửa được điều
này ngoài fold.

Theo loại hợp đồng, sau calibration: Cash loans 1.02 / 1.01, Revolving loans 0.94 / 1.00 (champion /
challenger). Không cần calibrate riêng theo phân khúc.

Với scorecard, bảng điểm không đổi, chỉ cách đọc điểm thành PD thay đổi: 600 điểm = PD 1.98% (thang gốc
1.96%), 550 điểm = 9.80% (gốc 10.16%), 500 điểm = 36.85% (gốc 39.02%).

## 5. Model package

`data/processed/home_credit/model_package/`:

| File | Nội dung |
|---|---|
| `champion.json` | Scorecard Stage 3 (bảng điểm, bin, hệ số) + calibrator |
| `challenger.json`, `challenger_model.txt` | LightGBM Stage 4 (cây, category, tham số) + calibrator |
| `calibrated_scores.parquet` | Score, PD đã calibrate của hai mô hình và mã lý do, cho mọi hồ sơ |
| `package.json` | Fingerprint gói, fingerprint hai mô hình, split, config hash, SHA-256 từng file |

Fingerprint gói tính từ fingerprint hai mô hình và tham số calibrator, không từ byte của file: chạy lại
bước train ra cùng mô hình với run_id mới không bị coi là gói mới. SHA-256 từng file vẫn được ghi để chặng
6 kiểm tra toàn vẹn.

## 6. Definition of Done: bằng chứng

`tests/test_stage5_dod.py` chạy chuỗi lệnh thật (ingest -> ... -> calibrate):

- cùng dữ liệu cho cùng fingerprint gói và cùng SHA-256 của `calibrated_scores.parquet`;
- sửa mọi hồ sơ **test**: fingerprint gói **không đổi**;
- sửa 1,500 hồ sơ **calibration**: hai mô hình **không đổi** (calibration không vào train), calibrator và
  fingerprint gói **đổi** (đối chứng: calibrator học đúng trên tập calibration);
- SHA-256 trong `package.json` khớp file; PD công bố khớp với việc áp calibrator của gói lên đầu ra mô hình;
- với Platt, điểm cao hơn luôn có PD thấp hơn.

`tests/test_calibration.py`: Platt khôi phục đúng một méo đã biết (slope 0.7 -> 1.0), isotonic sửa được
dạng lệch Platt không sửa được và được chọn khi đó, quy tắc chọn ưu tiên Platt khi chênh nhỏ, chỉ số của
một mô hình hoàn hảo đúng bằng lý thuyết.

## 7. Giới hạn

- PD khớp bad rate của dữ liệu phát triển (8.07%). Danh mục thật có bad rate khác thì cần hiệu chỉnh theo
  kỳ vọng dài hạn (central tendency), ngoài phạm vi project.
- Không có out-of-time: calibration đúng trên mẫu ngẫu nhiên cùng giai đoạn.
- Các nhóm PD cao có ít quan sát; tỷ lệ bad rate / PD ở đó dao động nhiều hơn.

## 8. Tái lập

```
python -m credit_risk run --steps calibrate --source home_credit
```

Vài giây, khi đã có đầu ra của bước train. Báo cáo: `artifacts/<run_id>/calibrate/calibration_report.md`.
