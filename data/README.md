# Dữ liệu

Repo **không chứa dữ liệu**. Mọi thứ trong `data/` (trừ file này) đều nằm trong `.gitignore`.
Điều khoản của cả hai nguồn không cho phép phân phối lại dữ liệu.

```
data/
├── raw/                      # file gốc, chỉ đọc, không bao giờ sửa
│   ├── home_credit/          # 8 file CSV của Kaggle
│   └── freddie_mac/          # sample_orig_YYYY.txt, sample_perf_YYYY.txt
└── processed/                # parquet do bước ingest sinh ra, kèm _ingest_manifest.json
```

Có thể để dữ liệu ở ổ khác bằng cách đặt `CREDIT_RISK_DATA_DIR` trong `.env`.

## 1. Home Credit Default Risk (Module 1-2)

1. Đăng nhập Kaggle, mở <https://www.kaggle.com/competitions/home-credit-default-risk/data>, bấm **Join / I Understand and Accept** để chấp nhận rules của cuộc thi.
2. Tải về bằng một trong hai cách:
   - Trình duyệt: **Download All**.
   - Kaggle CLI (cần API token trong `%USERPROFILE%\.kaggle\kaggle.json`):
     ```
     pip install kaggle
     kaggle competitions download -c home-credit-default-risk -p data/raw/home_credit
     ```
3. Giải nén vào `data/raw/home_credit/`. Cần có đủ 8 file sau (khoảng 2.5 GB sau giải nén):

| File | Dùng cho |
|---|---|
| `application_train.csv` | Population phát triển, có `TARGET` |
| `application_test.csv` | Không có nhãn, chỉ dùng làm "mẫu hiện tại" khi tính PSI |
| `bureau.csv`, `bureau_balance.csv` | Lịch sử tín dụng ở tổ chức khác |
| `previous_application.csv` | Các hồ sơ trước tại Home Credit |
| `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` | Hành vi trả nợ các khoản trước |

`HomeCredit_columns_description.csv` và `sample_submission.csv` không bắt buộc (file mô tả cột nên giữ để tra cứu).

## 2. Freddie Mac Single-Family Loan-Level Dataset, bản sample (Module 3)

Bản sample là mẫu ngẫu nhiên **50.000 khoản vay cho mỗi năm giải ngân**, đủ nhỏ cho laptop 8 GB RAM.

1. Đăng ký miễn phí và đăng nhập **Clarity Data Intelligence** qua trang dataset:
   <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>
2. Vào mục Data Download (<https://claritydownload.fmapps.freddiemac.com/CRT/>), tải `sample_YYYY.zip` cho các năm **2012-2019** (danh sách năm nằm trong `configs/ews.yaml`).
3. Giải nén để có `sample_orig_YYYY.txt` và `sample_perf_YYYY.txt`, đặt thẳng vào `data/raw/freddie_mac/` (không để thư mục con).

Lưu ý về file layout:

- File không có header. Thứ tự cột được khai báo trong `configs/data_contracts/freddie_mac.yaml` theo **File Layout áp dụng từ July 2026 (Release 47)**: origination 31 cột, performance 35 cột.
- Nếu Freddie Mac đổi layout, `ingest` sẽ dừng với thông báo số cột không khớp. Khi đó tải File Layout mới trên trang dataset và cập nhật contract.
- Mã quá hạn là `00`, `01`, `02`, `03`... (cap ở `99`), `RA` = REO acquisition, `XX` = không có dữ liệu.

Nên thử trước với 1-2 năm để kiểm tra pipeline, rồi mới tải đủ 8 năm.

## 3. Sau khi tải

```
python -m credit_risk run --steps ingest validate-data
```

Chỉ có một nguồn thì thêm `--source home_credit` hoặc `--source freddie_mac`.

Báo cáo nằm ở `artifacts/<run_id>/data_validation/<source>.md`. Ở lần chạy đầu trên dữ liệu thật:

1. Rule `error` fail thì pipeline dừng: đọc báo cáo, sửa nguyên nhân rồi chạy lại.
2. Rule `warn` là rule chưa được xác nhận trên dữ liệu thật. Rule nào đúng thì nâng lên `error` trong contract. Rule nào sai thì sửa lại và ghi lý do trong comment.
