# Stage 7: Validation report tự động (Home Credit)

Chặng 6 đo và liệt kê; chặng này **kết luận**. Cùng một lệnh `validate-model` giờ đi thêm bước cuối: đọc
`metrics.json` vừa tính, áp các quy tắc trong `validation.yaml -> conclusion_rules`, cho mỗi mô hình một kết
luận Đạt / Đạt có lưu ý / Phải khắc phục, rồi sinh khuyến nghị triển khai. Không câu kết luận nào viết tay:
đổi ngưỡng trong config thì kết luận đổi theo.

Con số dưới đây lấy từ lần chạy `20261025T064218Z-88c3b128` (chạy lại `train` + `calibrate` + `validate-model`
sau khi đổi cách băm ở chặng 6.0; gói `c95dc90913d8` tái lập đúng): **259 kiểm tra, 0 lệch, 17 phát hiện**.

> **Cách đọc nhanh:** mỗi quy tắc nhìn một con số trong `metrics.json`, so với ngưỡng trong config, và cho
> `pass` / `note` / `remediate` cho **từng** mô hình. Kết luận của mô hình là kết quả nặng nhất. Khuyến nghị
> (triển khai mô hình nào, giữ mô hình nào làm đối chứng) suy ra từ hai kết luận và chênh lệch Gini có cặp.

## 1. Kết luận nhanh

- **champion: Phải khắc phục.** Người dưới 30 tuổi được duyệt chỉ bằng 0.56 lần nhóm được duyệt nhiều nhất
  (AIR, ngưỡng khắc phục 0.60). Thêm ba nhóm tuổi lệch hiệu chỉnh và AUC nhóm 60+ thấp hơn tổng thể 0.085.
- **challenger: Đạt có lưu ý.** Không quy tắc nào ở mức khắc phục. Bốn lưu ý: `APP_ANNUITY_TO_CREDIT` trôi
  (PSI 1.00 trên application_test), AIR nhóm dưới 30 là 0.66, nhóm 45-60 bị PD thừa (z −3.2), AUC nhóm 60+
  thấp hơn tổng thể 0.057.
- **Khuyến nghị:** triển khai challenger kèm điều kiện, giữ champion làm đối chứng, không dùng champion ra
  quyết định. Challenger cũng tốt hơn có ý nghĩa trên test: Gini +0.058, khoảng [+0.049, +0.067].
- **Ở cùng số hồ sơ bị từ chối với champion (26.6%)**, nhóm được duyệt của challenger có bad rate 3.86% so với
  4.30%, chặn 65.0% số bad so với 61.0%.

## 2. Từ đo lường đến kết luận

| # | Phần | Làm gì |
|---|---|---|
| 7.0 | Sửa slope | `calibration_slope_intercept` dùng `tol=1e-10` (phát hiện 2 của chặng 6); validator tìm artifacts của `train` theo run của scorecard, của `calibrate` theo run của gói |
| 7.1 | `conclusion.py` | 11 quy tắc, ngưỡng nằm ở `validation.yaml -> conclusion_rules`; kết luận từng mô hình; khuyến nghị |
| 7.2 | `comparison.py` | So sánh hai mô hình cạnh nhau, mức đồng thuận xếp hạng, tập hoán đổi quyết định, so ở cùng số từ chối |
| 7.3 | Nối vào engine, báo cáo | Kết luận là bước cuối của `validate-model`; báo cáo mở đầu bằng kết luận |

Quy tắc chỉ đọc `metrics.json` (các bảng dạng records), không đọc dữ liệu hay gói. Vì vậy kết luận **tính
lại được từ riêng file `metrics.json`**, kể cả rất lâu sau lần chạy: `conclude(metrics, validation)` cho đúng
mục `conclusion` đã ghi (test).

### Các quy tắc và ngưỡng (quyết định 2026-10-25)

| Quy tắc | Nhìn vào | Lưu ý | Khắc phục |
|---|---|---|---|
| `blocking_checks` | kiểm tra lệch ở `package` / `population` / `implementation` | | ≥ 1 |
| `auc_ci_lower` | biên dưới khoảng 95% của AUC trên test | < 0.72 | < 0.70 |
| `validation_test_gap` | Gini validation − test | khoảng 95% không chứa 0 và > 0 | |
| `slope` | slope hiệu chỉnh trên test | ngoài [0.90, 1.10] hoặc \|z\| > 3 | ngoài [0.80, 1.20] |
| `bad_rate_to_pd_ratio` | bad rate / PD trung bình trên test | ngoài [0.95, 1.05] | |
| `flagged_bands` | số band PD lệch hệ thống | ≥ 3 | |
| `output_psi` | PSI score / PD so với application_test | vàng | đỏ |
| `feature_psi` | PSI 5 biến ảnh hưởng nhất | có biến đỏ | |
| `sensitive_calibration` | nhóm nhạy cảm (n ≥ 500) bị cờ hiệu chỉnh | ≥ 1 nhóm | |
| `adverse_impact_ratio` | AIR thấp nhất trong các nhóm nhạy cảm | < 0.80 | < 0.60 |
| `group_auc_gap` | AUC nhóm nhạy cảm trừ AUC tổng thể | < −0.05 | |

Khuyến nghị: nếu cả hai đều Phải khắc phục thì không triển khai mô hình nào; nếu chỉ một mô hình đủ điều kiện
thì triển khai mô hình đó; nếu cả hai đủ điều kiện thì chọn mô hình có Gini cao hơn có ý nghĩa, không có thì
chọn champion vì đơn giản và giải thích được. Các lưu ý của mô hình được chọn trở thành **điều kiện kèm theo**.

## 3. Kết quả trên dữ liệu thật

| Quy tắc | champion | challenger |
|---|---|---|
| `blocking_checks` | đạt (0 lệch) | đạt (0 lệch) |
| `auc_ci_lower` | đạt (0.7523, biên dưới 0.7438) | đạt (0.7813, biên dưới 0.7735) |
| `validation_test_gap` | đạt (+0.011 [−0.015, +0.035]) | đạt (+0.017 [−0.007, +0.039]) |
| `slope` | đạt (0.964, \|z\| 1.9) | đạt (0.971, \|z\| 1.7) |
| `bad_rate_to_pd_ratio` | đạt (0.999) | đạt (1.000) |
| `flagged_bands` | đạt (2 band) | đạt (1 band) |
| `output_psi` | đạt (0.0013) | đạt (0.0026) |
| `feature_psi` | đạt (lớn nhất `EXT_SOURCE_1` 0.082) | **lưu ý** (`APP_ANNUITY_TO_CREDIT` 1.00) |
| `sensitive_calibration` | **lưu ý** (3 / 11 nhóm, đều là nhóm tuổi) | **lưu ý** (1 / 11: 45-60) |
| `adverse_impact_ratio` | **khắc phục** (0.56, dưới 30) | **lưu ý** (0.66, dưới 30) |
| `group_auc_gap` | **lưu ý** (−0.085, 60+) | **lưu ý** (−0.057, 60+) |

Điểm đáng chú ý nhất cho người phê duyệt: **các vấn đề của cả hai mô hình đều nằm ở tuổi**. Đó là hệ quả trực
tiếp của quyết định giữ `DAYS_BIRTH`, `CODE_GENDER`, `NAME_FAMILY_STATUS` theo tiêu chí thống kê
(PROJECT_SCOPE #7). Challenger nhẹ hơn ở mọi quy tắc công bằng, nhưng vẫn dưới mốc 4/5 cho người dưới 30.

## 4. Báo cáo

`artifacts/<run_id>/validate-model/validation_report.md` giờ đọc theo thứ tự:

1. Tiêu đề và provenance (fingerprint gói, champion, challenger, split; run đóng băng; config hash; seed; số
   lần bootstrap), một dòng tổng: số kiểm tra, số phát hiện, **kết luận hai mô hình**.
2. **Kết luận:** kết luận từng mô hình, khuyến nghị, bảng 11 quy tắc × 2 mô hình (giá trị, ngưỡng, kết quả,
   diễn giải).
3. **Con số chính (test):** AUC / Gini / KS kèm khoảng 95%, slope, bad rate / PD, PSI đầu ra.
4. **Việc cần làm:** cho từng mô hình, mục khắc phục trước rồi mục lưu ý, mỗi mục kèm câu khuyến nghị.
5. **Phạm vi và giới hạn:** nguyên văn `validation.yaml -> report.limitations` (6 dòng: không có
   out-of-time, application_test không có TARGET, PSI tách theo `NAME_CONTRACT_TYPE`, bootstrap chưa hiệu
   chỉnh đa so sánh, thuộc tính nhạy cảm, quy tắc từ chối dùng chung), cộng những gì lần chạy tự phát hiện:
   phân khúc thiếu cột, không thấy `splits_summary.json`, artifacts của run `train` / `calibrate` không còn
   trên máy. Lần chạy thật không có giới hạn tự phát hiện nào.
6. Kiểm tra theo module, phát hiện, kiểm tra lệch, rồi chi tiết từng module như chặng 6 (thêm phần so sánh
   hai mô hình của 7.2).

`metrics.json` thêm `conclusion` (`verdicts`, `rules`, `recommendation`) và `limitations`; `provenance` thêm
`verdicts`, `deploy`, `headline_sample`. Dòng tóm tắt của bước trong `run_manifest.json` ghi kết luận hai mô
hình và mô hình được khuyến nghị.

## 5. Definition of Done: bằng chứng

`tests/test_stage7_dod.py` chạy bước thật qua pipeline trên gói tổng hợp (fixture `stage6_build`):

- một lệnh cho kết luận trong `metrics.json`, đủ các quy tắc trong config; `conclude()` trên riêng
  `metrics.json` cho đúng kết luận đã ghi; provenance và manifest khớp kết luận;
- báo cáo đúng thứ tự Kết luận → Con số chính → Việc cần làm → Phạm vi và giới hạn → kiểm tra → phát hiện →
  module; mọi quy tắc lưu ý / khắc phục có mặt trong "Việc cần làm";
- **ngưỡng quyết định kết luận:** ngưỡng AUC không thể đạt cho cả hai Phải khắc phục và "Không mô hình nào";
  chỉ giữ `blocking_checks` cho cả hai Đạt;
- giới hạn in nguyên văn từ config, thay config thì báo cáo đổi theo;
- thiếu `conclusion_rules`: bước thất bại, manifest nêu rõ lý do.

Test module: `tests/test_validation_conclusion.py` (22 test: mỗi quy tắc lật đúng ở ngưỡng và chỉ cho đúng
mô hình, bốn nhánh khuyến nghị, lỗi config), `tests/test_validation_comparison.py` (6 test).

## 6. Giới hạn

- Ngưỡng là quyết định của dự án (2026-10-25), không phải chuẩn của cơ quan quản lý. Mốc 4/5 cho AIR mượn từ
  thực hành công bằng tín dụng ở Mỹ.
- Kết luận chỉ phủ những gì `metrics.json` đo được; mọi giới hạn của chặng 6 vẫn còn (không có out-of-time,
  application_test không có nhãn, chưa hiệu chỉnh đa so sánh).
- Khuyến nghị chỉ chọn giữa hai mô hình đã có; không đề xuất mô hình mới hay cách sửa cụ thể.

## 7. Tái lập

```
python -m credit_risk run --steps validate-model --source home_credit
```

Khoảng 15-30 giây khi đã có gói Stage 5. Tính lại kết luận từ một `metrics.json` cũ với ngưỡng hiện tại:

```python
import json
from credit_risk.config import load_config
from credit_risk.settings import load_settings
from credit_risk.validation.conclusion import conclude, conclusion_markdown

settings = load_settings()
metrics = json.loads(open("artifacts/<run_id>/validate-model/metrics.json", encoding="utf-8").read())
print(conclusion_markdown(conclude(metrics, load_config(settings.config_dir).validation)))
```
