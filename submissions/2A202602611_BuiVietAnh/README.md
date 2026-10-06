# Bài nộp Lab Day 2 — DeepWeeds

- **Họ và tên:** Bùi Việt Anh
- **Mã sinh viên:** 2A202602611
- **Notebook Kaggle:** [lap-d2p2](https://www.kaggle.com/code/frostyashe/lap-d2p2)
- **Notebook trong repo:** [code/lab_day2.ipynb](code/lab_day2.ipynb)

## Cách chạy lại

Mở notebook Kaggle ở link trên và chọn **Run All** để chạy toàn bộ bài. Môi trường cần bật **GPU** và **Internet** để tải dữ liệu, thư viện và trọng số mô hình. Notebook đã có các ô chuẩn bị cần thiết, không cần chạy lệnh riêng.

Các ô chạy lần lượt: chuẩn bị môi trường và tải dữ liệu → kiểm tra dữ liệu và quá trình huấn luyện → so sánh 5 mô hình → so sánh công thức huấn luyện → so sánh cách dự đoán → đánh giá cấu hình cuối với 3 seed → tính lại chỉ số bằng `eval.py` → xuất bảng kết quả, báo cáo và bộ file nộp.

## Môi trường và thư viện

Phiên bản được ghi trong kết quả của notebook:

| Thành phần | Phiên bản / thiết bị |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| timm | 1.0.29 |
| GPU | Tesla T4 |

Notebook dùng thêm `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, `openpyxl` và `fvcore`. Ô cài đặt chạy `pip -q install timm openpyxl fvcore`; các thư viện này không được cố định phiên bản trong lệnh cài đặt. Phiên bản Python, PyTorch, timm và tên GPU thực tế được in khi khởi động notebook.

## Dữ liệu và seed

Dùng bộ ảnh **DeepWeeds**, gồm 17.509 ảnh thuộc 9 lớp. Giữ nguyên cách chia **fold 0** qua `train_subset0.csv`, `val_subset0.csv` và `test_subset0.csv`. Tập val dùng để chọn cấu hình; tập test dùng để báo cáo kết quả cuối.

Seed là số dùng để cố định các thao tác ngẫu nhiên:

- Kiểm tra dữ liệu và huấn luyện trên một nhóm ảnh nhỏ: **42**.
- So sánh mô hình và công thức huấn luyện: **0**.
- Đánh giá cấu hình cuối `F01` và mốc so sánh `T00`: **0, 1, 2**; báo cáo trung bình và độ lệch chuẩn qua 3 lần chạy.

Seed không thay đổi cách chia dữ liệu. Mỗi cấu hình cuối/seed dự đoán test một lượt; các bước tổng hợp sau đó đọc file dự đoán đã lưu.

## Các file chính

| File / thư mục | Nội dung |
|---|---|
| [code/](code/) | Notebook và mã nguồn xử lý dữ liệu, mô hình, huấn luyện, dự đoán, đo tốc độ và tổng hợp kết quả |
| [code/results.xlsx](code/results.xlsx) | Bảng kết quả đang lưu trong repo |
| [code/report.md](code/report.md) | Báo cáo đang lưu trong repo |
| [code/curves/](code/curves/) | Biểu đồ đang lưu trong repo |
| [code/predictions/](code/predictions/) | Các file dự đoán đang lưu trong repo |

Sau khi chạy xong bước xuất sản phẩm, notebook tạo `results.xlsx`, `report.md`, `curves/`, `predictions/`, `evidence/`, `eval_out/` và `submission_bundle.zip` ngay trong thư mục bài nộp. Cấu hình và phiên bản PyTorch/timm của từng lần huấn luyện được lưu tại `evidence/runs/<exp_id>/seed<k>/config.json`.

Dữ liệu ảnh và file trọng số mô hình lớn không đưa vào bộ file nộp. Công cụ đánh giá dùng [eval.py gốc](../../eval.py).
