# Bài viết tổng kết: một vòng đời mô hình rủi ro tín dụng, từ định nghĩa đến watchlist

Project này trả lời một câu hỏi đơn giản: *nếu phải dựng trọn vòng đời của một mô hình rủi ro tín dụng trong ngân
hàng, bằng dữ liệu công khai, trên một laptop, thì nó trông như thế nào khi làm đúng kỷ luật?* Không phải một
notebook Kaggle, không phải một hệ thống hạ tầng; mà là bốn việc ngân hàng thật sự phải làm, nối với nhau bằng một
pipeline chạy từ config, với mỗi con số truy ngược được về nơi sinh ra nó.

```
Model Development  ->  Independent Validation  ->  Portfolio Monitoring  ->  Early Warning
   (Home Credit)            (Home Credit)                (Freddie Mac)            (Freddie Mac)
```

## 1. Định nghĩa trước, mô hình sau

Việc đầu tiên không phải là fit một mô hình mà là viết `configs/definitions.yaml`: 90+ DPD là gì, cửa sổ ba tháng
tính từ đâu, cure rồi tái vỡ nợ có tính là sự kiện không (có), COVID 2020-03 → 2021-12 là stress segment, thang điểm
600 tại odds 50:1 với PDO 20. Mọi chỗ trong code đọc từ file này; đổi một định nghĩa là đổi config hash ghi trong
manifest của mỗi lần chạy.

Bên cạnh đó là data contract cho từng nguồn (kiểu, khoá, miền giá trị, mức khuyết cho phép) và bước `validate-data`
trả về đạt / không đạt trước khi bất kỳ bước nào khác được chạy. Trên dữ liệu thật: Home Credit 135 kiểm tra, 0 lỗi;
Freddie Mac 14 lứa 2012-2025, 700k khoản, 35.6 triệu dòng tháng, 0 lỗi. Những gì học được từ dữ liệu ghi trong
`docs/data_quality_*.md`: 31.8% khoản Freddie là HARP với DTI cấu trúc null, 13.4% khoản vào panel khi đã có tuổi
(left truncation), tỷ lệ 90+ thời COVID gấp 3-4 lần trước đó.

## 2. Module 1: scorecard giải thích được, challenger để so

Home Credit không có ngày nộp đơn, nên split là stratified random 60/10/15/15 và **không có out-of-time thật**;
điều này được ghi vào model card và báo cáo thẩm định thay vì giấu đi. Trước khi chọn biến có một bảng *feature
availability*: mỗi cột có mặt từ lúc nào (tại đơn, lịch sử, sau quyết định), cột không khớp quy tắc làm bước dừng.
Đó là control chống leakage rẻ nhất và hiệu quả nhất.

Binning tự viết (không dùng optbinning), fit chỉ trên train, đơn điệu trừ hai ngoại lệ được ghi nhận; WoE / IV trên
train, kiểm ổn định trên validation; chọn 45 biến theo IV → cờ ngoài mẫu → tương quan → VIF. Scorecard là logistic
trên WoE với vòng kiểm dấu hệ số bằng bootstrap: 13 biến bị loại vì hệ số dương hoặc không ổn định, còn 32. Điểm cơ
sở 557, Gini validation 0.515, mỗi hồ sơ có 4 mã lý do tính từ số điểm mất so với bin tốt nhất.

LightGBM challenger dùng đúng tập dân số và split đó, 30 cấu hình có log, early stopping, SHAP lấy từ chính LightGBM.
Gini validation 0.580, hơn champion 0.064 nhưng con số này lạc quan vì chọn cấu hình trên validation. Cả hai được
calibrate bằng Platt (chọn qua CV) rồi đóng băng thành một *model package* có SHA-256 và fingerprint.

Một quyết định có chủ đích: giới tính, tình trạng hôn nhân và tuổi **không** bị loại theo chính sách. Hệ quả được
ghi ngay trong PROJECT_SCOPE: model card phải nêu rủi ro fairness, và Module 2 bắt buộc phân tích theo ba nhóm đó.

## 3. Module 2: thẩm định độc lập, kết luận tự động

Engine thẩm định không tin bất kỳ con số nào của bước phát triển. Nó đọc gói đóng băng, kiểm tra 15 điều về tính toàn
vẹn và danh tính (SHA-256, fingerprint tính lại), đếm lại dân số, rồi tính lại AUC / Gini / KS, hiệu chỉnh
(slope, intercept, Hosmer-Lemeshow, band), PSI với ranh giới bin cố định từ train, bootstrap 1,000 lần có ghép cặp,
phân khúc, so sánh hai mô hình (swap set, cùng khối lượng từ chối), và chấm lại 5,000 hồ sơ từ đầu. 259 kiểm tra, 0
lệch, 17 phát hiện.

Chặng 7 biến phát hiện thành kết luận: 11 quy tắc trong `validation.yaml` (kiểm tra lệch, cận dưới CI của AUC, chênh
validation-test, slope, PSI output và feature, hiệu chỉnh và AIR theo nhóm nhạy cảm, chênh AUC theo nhóm) cho ra
Đạt / Đạt có lưu ý / Phải khắc phục. Trên dữ liệu thật: **champion Phải khắc phục** vì adverse impact ratio 0.56 cho
người dưới 30 tuổi; **challenger Đạt có lưu ý**; khuyến nghị triển khai challenger kèm điều kiện. Test Gini 0.505
[0.488, 0.520] so với 0.563 [0.547, 0.578]; chênh +0.058 [+0.049, +0.067] khác 0. Ở cùng khối lượng từ chối,
challenger cho bad rate duyệt 3.86% thay vì 4.30%.

Báo cáo mở đầu bằng kết luận, việc cần làm và giới hạn, rồi mới đến bảng. Không chỉnh tay.

## 4. Module 3: danh mục thế chấp và cảnh báo sớm

Panel 35.6 triệu dòng được dựng trong DuckDB (laptop 7.4 GB RAM, giới hạn 3 GB), với bucket trạng thái và chuyển
nhóm có bản pandas tham chiếu để đối chiếu trên mẫu. Chỉ số theo số khoản và dư nợ, vintage curve theo lứa, roll rate,
ma trận dịch chuyển theo giai đoạn (trước / trong / sau COVID), tập trung rủi ro. Cảnh báo so với 12 tháng trước và
so lứa với lứa ở cùng MOB. Tháng 2026-03: chỉ số tháng xanh nhưng trạng thái **đỏ**, vì lứa 2023-2024 có default
tích luỹ ở cùng MOB cao gấp khoảng 3 lần lứa 2012-2016. Đổi tháng báo cáo trong config là mọi bảng tự làm mới.

EWS gán nhãn *lần đầu vào 90+ hoặc thanh lý trong 3 tháng tới* trên toàn panel bằng SQL, khớp tuyệt đối với bản
pandas có test; khoản đã 90+ hoặc đã tất toán bị loại khỏi cohort. 34 feature tại t chỉ từ dòng ≤ t, có test "sửa
tháng sau t thì feature tại t không đổi". Dữ liệu chia theo thời gian: train → validation → stress test COVID →
out-of-time 2022-2025; lứa 2020-2025 không bao giờ vào train. Ngưỡng cảnh báo là công suất xử lý (top 5% mỗi tháng),
không phải 0.5; chỉ số là PR-AUC, capture, lift. Out-of-time: watchlist bắt được 90.3% sự kiện (lift 18), và 51.5% trong
nhóm khó nhất là khoản đang current. Watchlist 2026-03: 16,159 khoản kèm reason code. Survival: Kaplan-Meier và
Aalen-Johansen với trả trước là rủi ro cạnh tranh; CIF default MOB 60 là 3.3%, trả trước 46.8%.

## 5. Chặng 10: để người lạ hiểu trong mười phút

Bước `report` gom output mới nhất của mọi bước thành một bundle: `index.json` ghi run id của từng phần và các con số
chính, kèm báo cáo, bảng và 12 hồ sơ mẫu với điểm từng biến. Train và calibrate được lấy đúng theo run mà
validate-model đã thẩm định, nên dashboard không bao giờ ghép mô hình của lần này với kết luận của lần khác.
Dashboard Streamlit bảy trang chỉ đọc bundle đó, theo luồng quyết định: vấn đề → một hồ sơ cụ thể → kết luận phê
duyệt → lứa đang xấu đi → watchlist → kỹ thuật → giới hạn.

## 6. Những điều học được (không có trong sách)

- **Số thực cộng không giao hoán.** DuckDB cộng song song cho kết quả khác nhau ở bit cuối giữa các lần chạy, đủ để
  đổi fingerprint. Cách sửa: tổng và trung bình bằng `DECIMAL(38,10)` khi cần tái lập.
- **Fingerprint phụ thuộc máy.** lbfgs hội tụ cùng một nghiệm đến 1e-12 nhưng không cùng bit; băm số thực đã làm
  tròn 8 chữ số.
- **`GROUP BY` không có thứ tự.** Một test "chạy hai lần cho cùng kết quả" thất bại ngẫu nhiên chỉ vì thứ tự dòng.
- **Kiểm exit code của pytest, không phải của `tail`.** Một commit từng được đẩy lên với test đang hỏng vì lỗi này.
- **Left truncation và HARP là chuyện của dữ liệu, không phải lỗi.** Ghi nhận, đo ảnh hưởng (bỏ khoản bị cắt trái
  thì CIF gần như không đổi), rồi đi tiếp.
- **Kết luận phải có quy tắc viết ra.** Chừng nào "Đạt" còn là nhận xét của người, nó chưa phải là thẩm định.

## 7. Giới hạn và hướng đi

Không có out-of-time thật cho Module 1; champion cần khắc phục fairness trước khi dùng; EWS chưa có biến vĩ mô và
chưa hiệu chỉnh lại PD theo đoạn; survival mới mô tả, chưa có covariate; dữ liệu Freddie là mẫu 50,000 khoản mỗi
lứa. Cố tình không làm: LGD / EAD / IFRS 9, hệ thống phê duyệt thật, Spark / Airflow / MLflow. Phần lõi ổn định rồi
thì mới mở rộng, theo đúng thứ tự ưu tiên đã viết trong OVERVIEW từ ngày đầu.

Toàn bộ: 9 bước pipeline, 10 chặng có Definition of Done, 424 test, 7 trang dashboard. Mỗi chặng có một file
`docs/stage*.md` ghi quyết định, kết quả trên dữ liệu thật và bằng chứng hoàn thành.
