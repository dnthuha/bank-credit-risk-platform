# Data quality: Home Credit Default Risk

Profile lần đầu trên dữ liệu thật, ngày 2026-09-18. Tài liệu này là đầu vào cho
Stage 2 (feature availability matrix, binning) và Module 2 (PSI).

- Ingest: run `20260918T151537Z-b0481463`, 8 bảng, 191 giây, 2.6 GB CSV thành 532 MB parquet.
- Validate sau khi chỉnh contract: run `20260918T155225Z-adf64872`, **135 check, 0 error, 1 warning** (vấn đề đã biết, mục 2).
- Contract: [`configs/data_contracts/home_credit.yaml`](../configs/data_contracts/home_credit.yaml)

Tái lập:

```
python -m credit_risk run --steps ingest validate-data --source home_credit
```

## 1. Quy mô

| Bảng | Số dòng | Khoá |
|---|---:|---|
| application_train | 307,511 | SK_ID_CURR |
| application_test | 48,744 | SK_ID_CURR |
| bureau | 1,716,428 | SK_ID_BUREAU |
| bureau_balance | 27,299,925 | SK_ID_BUREAU + MONTHS_BALANCE |
| previous_application | 1,670,214 | SK_ID_PREV |
| POS_CASH_balance | 10,001,358 | SK_ID_PREV + MONTHS_BALANCE |
| credit_card_balance | 3,840,312 | SK_ID_PREV + MONTHS_BALANCE |
| installments_payments | 13,605,401 | không có khoá duy nhất |

Bad rate của train: **8.07%** (24,825 bad / 307,511).

## 2. Vấn đề đã biết của dữ liệu gốc

Những điểm này có sẵn trong dữ liệu gốc, không sửa được ở nguồn. Chúng được xử lý ở bước `features`.

| Vấn đề | Quy mô | Rủi ro | Xử lý |
|---|---|---|---|
| `bureau.DAYS_CREDIT_UPDATE` > 0 | 17 dòng, tối đa +372 ngày | Thông tin cập nhật **sau** ngày nộp đơn, tức là leakage | Rule `warn` trong contract; loại hoặc clip về 0 khi tạo feature |
| `SK_ID_PREV` không có trong previous_application | POS 340,561 dòng (3.4%), credit card 1,082,816 (28%), installments 1,250,826 (9.2%) | Join qua previous_application sẽ mất các dòng này | Tổng hợp **trực tiếp theo `SK_ID_CURR`** (khớp 100%) |
| `SK_ID_BUREAU` của bureau_balance không có trong bureau | 3,120,184 dòng (11.4%) | Bảng không có `SK_ID_CURR`, nên các dòng này không gắn được với hồ sơ nào | Bỏ qua khi tổng hợp; ghi nhận là giới hạn |
| Ngày bất khả thi trong bureau | `DAYS_CREDIT_ENDDATE`, `DAYS_ENDDATE_FACT`, `DAYS_CREDIT_UPDATE` cũ tới -42,060 ngày (khoảng 115 năm), trong khi `DAYS_CREDIT` >= -2,922 | Làm lệch các feature thời lượng | Clip về miền hợp lý khi tạo feature |
| Thiếu bản ghi thanh toán | 2,905 dòng installments thiếu cùng lúc `DAYS_ENTRY_PAYMENT` và `AMT_PAYMENT` | Không biết kỳ đó đã trả hay chưa | Tách thành cờ riêng, không coi là trả 0 |

Các giá trị **dương nhưng hợp lệ** (không có rule `max: 0`):

- `bureau.DAYS_CREDIT_ENDDATE` > 0 (602,603 dòng): thời hạn còn lại theo hợp đồng, đã biết lúc nộp đơn.
- `previous_application.DAYS_LAST_DUE_1ST_VERSION` > 0 (224,392 dòng): ngày đáo hạn dự kiến theo hợp đồng cũ.

## 3. Phát hiện cho Stage 2 (features, binning)

1. **`DAYS_EMPLOYED = 365243`** xuất hiện ở 55,374 dòng (18%): 55,352 Pensioner và 22 Unemployed. `ORGANIZATION_TYPE = XNA` có đúng cùng số dòng. Đây là **một nhóm khách riêng**, không phải missing, nên cần bin riêng.
2. **Sentinel `365243` trong previous_application**: `DAYS_FIRST_DRAWING` 934,444 dòng, `DAYS_TERMINATION` 225,913, `DAYS_LAST_DUE` 211,221, `DAYS_LAST_DUE_1ST_VERSION` 93,864, `DAYS_FIRST_DUE` 40,645. Phải thay bằng null (kèm cờ) trước khi tổng hợp, nếu không min / max / mean sẽ vô nghĩa.
3. **Không có lịch sử khác với lịch sử bằng 0**: 44,020 hồ sơ train (14.3%) không có bản ghi bureau; 16,454 (5.4%) không có previous application. Cho vào bin "no history" riêng, không điền 0.
4. **Missing có ý nghĩa**: `EXT_SOURCE_1` thiếu 56.4%, `EXT_SOURCE_3` 19.8%, `EXT_SOURCE_2` 0.2%. Giữ bin missing riêng (`missing_as_own_bin: true`).
5. **Outlier**: `AMT_INCOME_TOTAL` lớn nhất là 117,000,000 (1 dòng, TARGET = 1), lớn thứ hai là 18,000,090. Chia bin theo phân vị chịu được; không dùng mean / std của biến này.
6. **Hạng mục hiếm**: `CODE_GENDER = XNA` có 4 dòng, `NAME_FAMILY_STATUS = Unknown` có 2 dòng. Gộp theo `rare_category_share`.
7. **Giá trị âm hợp lệ**: `bureau.AMT_CREDIT_SUM_DEBT` < 0 (8,418 dòng) và `credit_card_balance.AMT_BALANCE` < 0 (2,345 dòng). Nhiều khả năng là trả dư; cần xem trước khi tổng hợp.

## 4. Phát hiện cho Module 2 (PSI)

**Train và `application_test` có population khác nhau:**

| NAME_CONTRACT_TYPE | train | application_test |
|---|---:|---:|
| Cash loans | 90.5% | 99.1% |
| Revolving loans | 9.5% | 0.9% |

`application_test` được chọn là "mẫu hiện tại" khi tính PSI (PROJECT_SCOPE, quyết định #1). Vì vậy PSI của score và của các biến liên quan tới loại hợp đồng sẽ phản ánh cả **khác biệt do cách Kaggle chọn mẫu**, chứ không chỉ do thời gian. Validation report cần:

- báo cáo PSI tổng thể **và** PSI tách theo `NAME_CONTRACT_TYPE`;
- nói rõ nguyên nhân này khi PSI vượt ngưỡng, thay vì kết luận model không ổn định.
