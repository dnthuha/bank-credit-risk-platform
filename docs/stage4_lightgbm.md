# Stage 4: LightGBM challenger (Home Credit)

Đây là mô hình LightGBM được xây dựng để thách thức scorecard champion. Mục tiêu không chỉ là tìm chỉ số
cao hơn, mà còn đo rõ phần hiệu năng đánh đổi lấy sự phức tạp và khả năng giải thích. Tài liệu bao gồm
tuning có log, giải thích bằng SHAP và so sánh trực tiếp với champion. Mọi con số dưới đây lấy từ lần chạy
chính thức `20260925T042918Z-e6ed0154` (commit `085d86b`, working tree sạch), trên đúng split / binning của Stage 2 (fingerprint `bd8d70cd52bf` /
`27c1d5b79454`) và champion của Stage 3 (`a254458deddf`).

> **Cách đọc nhanh:** challenger là phương án thay thế để so sánh, chưa phải mô hình được đưa vào sử dụng.
> SHAP cho biết biến nào đẩy rủi ro của một hồ sơ lên hoặc xuống so với mức trung bình; nó khác với mã lý
> do theo điểm của scorecard.

## 1. Kết luận nhanh

- Gini validation **0.5795** (AUC 0.7898, KS 0.441), cao hơn champion **+0.064 Gini**. Tuy nhiên, đây vẫn
  là ước lượng lạc quan vì validation đã được dùng để dừng boosting và chọn cấu hình (mục 3); so sánh công
  bằng sẽ thực hiện ở chặng 6.
- 30 cấu hình cách nhau chỉ 0.0049 AUC validation: kết quả gần như không phụ thuộc cấu hình.
- Cấu hình chọn: 15 lá, `min_child_samples` 400, `feature_fraction` 0.5, 670 cây. Chênh AUC train -
  validation 0.064, so với 0.0055 của scorecard.
- Hai mô hình xếp hạng khá giống nhau (Spearman 0.875); 69% trong 10% hồ sơ rủi ro nhất là chung.
- `CODE_GENDER` đứng **thứ 4 / 192** theo SHAP và là mã lý do số 1 của 7.9% hồ sơ validation (mục 5).
- PSI của PD so với application_test: 0.0026 (Revolving loans 0.1151).
- Tái lập được: ba lần chạy (`20260925T030530Z`, `20260925T031425Z` trong lúc phát triển và
  `20260925T042918Z` từ commit sạch `085d86b`) cho cùng fingerprint challenger `78b3844ac54f`; champion vẫn
  là `a254458deddf`.

## 2. Thiết kế mô hình

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Đầu vào | Cả 192 biến của bảng feature Stage 2.3, giá trị gốc | Availability matrix đã chặn leakage; challenger cần thấy thứ scorecard bỏ để đo phần hiệu năng bị đánh đổi |
| Categorical | Tách native, category đóng băng từ train; category lạ -> missing | 16 biến; không cần WoE hay one-hot |
| Tuning | 30 cấu hình lấy ngẫu nhiên (seed 42) từ lưới 324 điểm; early stopping 100 vòng trên AUC validation | `max_trials: 30`, mọi cấu hình được ghi lại |
| Chọn cấu hình | Trong các cấu hình cách AUC tốt nhất <= 0.001, lấy cấu hình ít overfit nhất | Mục 3 |
| Giải thích | TreeSHAP của chính LightGBM (`pred_contrib`) | Không cần thêm gói `shap`; đóng góp cộng đúng bằng log-odds |
| Tái lập | `deterministic`, histogram theo cột, 4 luồng cố định | Cùng dữ liệu, cùng cấu hình cho cùng cây trên mọi máy |

## 3. Cách chọn cấu hình

Nếu chỉ chọn AUC validation cao nhất, trial 9 sẽ thắng: AUC 0.7903, **1,295 cây**, chênh train - validation
0.094. Trial 3 chỉ kém 0.0004 AUC nhưng có chênh 0.065. Sai số chuẩn của AUC trên khoảng 30k hồ sơ
validation vào khoảng 0.003, nên khác biệt 0.0004 nhỏ hơn nhiều mức có thể tin cậy và lại ưu ái mô hình
overfit hơn.

Quy tắc hiện tại (`model_dev.yaml` -> `challenger.selection_tolerance_auc: 0.001`): các cấu hình cách cấu
hình tốt nhất không quá 0.001 AUC coi như bằng nhau; trong nhóm đó chọn cấu hình có chênh train -
validation nhỏ nhất. Bốn cấu hình nằm trong ngưỡng; trial 3 được chọn:

| | Trial 9 (AUC cao nhất) | Trial 3 (được chọn) |
|---|---:|---:|
| AUC validation | 0.7903 | 0.7898 |
| Số cây | 1295 | 670 |
| Chênh train - validation | 0.094 | 0.064 |
| `num_leaves` / `min_child_samples` / `lambda_l2` | 15 / 100 / 10 | 15 / 400 / 1 |

## 4. So với champion (train và validation)

| Mẫu | Mô hình | AUC | Gini | KS |
|---|---|---:|---:|---:|
| train | champion | 0.7632 | 0.5264 | 0.3930 |
| train | challenger | 0.8538 | 0.7076 | 0.5446 |
| validation | champion | 0.7576 | 0.5153 | 0.3882 |
| validation | challenger | 0.7898 | 0.5795 | 0.4413 |

Phần hơn +0.064 Gini gồm hai nguồn trộn lẫn: mô hình phi tuyến, và thêm biến (160 biến scorecard không
dùng, ví dụ `POS_INSTALMENTS_LEFT_MEAN` hạng 6, `APP_ANNUITY_TO_CREDIT` hạng 9). Stage này không tách hai
phần đó.

Trong nhóm 10% rủi ro nhất, phần hai mô hình không trùng nhau cho thấy challenger chọn đúng hơn: hồ sơ chỉ
challenger xếp vào có bad rate 22.5%, hồ sơ chỉ champion xếp vào 14.6%.

## 5. Giải thích kết quả và thuộc tính nhạy cảm

- Ba biến dẫn đầu vẫn là `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, như scorecard. 26 trong 192 biến
  không được dùng trong cây nào.
- **`CODE_GENDER` hạng 4 / 192** theo trung bình |SHAP|, cao hơn nhiều so với vị trí trong scorecard (hạng
  14 theo độ rộng khoảng điểm). Nó là mã lý do số 1 của 7.9% hồ sơ validation. `NAME_FAMILY_STATUS` hạng
  21, `DAYS_BIRTH` hạng 37 (đã bị loại khỏi scorecard, challenger vẫn dùng).
- Mã lý do số 1 của hai mô hình trùng nhau ở 37% hồ sơ validation: champion đo "mất bao nhiêu điểm so với
  bin tốt nhất", challenger đo "đẩy rủi ro lên bao nhiêu so với trung bình mẫu". Không nên trộn hai loại
  mã lý do trong một báo cáo cho khách hàng.
- Báo cáo có 5 ví dụ giải thích từng hồ sơ (dạng waterfall, dưới dạng bảng) trải đều theo thứ hạng PD.

## 6. Bằng chứng hoàn thành

`tests/test_stage4_dod.py` chạy chuỗi lệnh thật (ingest -> validate-data -> features -> train):

- cùng dữ liệu cho cùng fingerprint challenger;
- sửa mọi hồ sơ **calibration và test**: fingerprint **không đổi**; sửa 300 hồ sơ train: **đổi** (đối chứng);
- PD và mã lý do trong `challenger_scores.parquet` khớp với việc chấm lại từ `challenger.json` +
  `challenger_model.txt`;
- `challenger.json` ghi fingerprint của binning và champion; manifest ghi phiên bản LightGBM.

`tests/test_challenger.py` kiểm tra: danh sách cấu hình cố định theo seed, quy tắc chọn (ngưỡng 0 = AUC cao
nhất; ngưỡng rộng = ít overfit nhất), category lạ thành missing, đóng góp SHAP cộng đúng bằng log-odds,
mã lý do, và fit lặp lại cho cùng cây.

Để test pipeline chạy nhanh, fixture của test dùng bản sao config với 3 cấu hình thay vì 30; search đầy
đủ được kiểm ở `test_challenger.py` và bằng lần chạy thật.

## 7. Việc cần theo dõi ở các chặng sau

- **Calibration (chặng 5)**: PD của challenger là đầu ra thô của LightGBM; chặng 5 calibrate cả hai mô hình.
- **So sánh công bằng (chặng 6)**: trên test, có bootstrap cho khoảng tin cậy của chênh Gini.
- **Fairness**: `CODE_GENDER` quan trọng hơn hẳn trong challenger. Module 2 cần báo cáo theo giới tính
  cho cả hai mô hình; có thể thử thêm một challenger không có ba thuộc tính nhạy cảm để đo giá phải trả.
- **Revolving loans**: PSI của PD 0.115, như scorecard, cần báo cáo riêng ở chặng 6.

## 8. Tái lập

```
python -m credit_risk run --steps train --source home_credit
```

Bước train chạy cả champion và challenger (khoảng 9 phút trên máy 4 luồng trở lên). Kết quả ở
`artifacts/<run_id>/train/`: `challenger_report.md`, `challenger_trials.csv`, `challenger_importance.csv`,
`challenger_performance.json`.
