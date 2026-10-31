# Cập nhật dữ liệu huấn luyện (TP. Hồ Chí Minh)

## Phạm vi

- Chỉ huấn luyện **giá bán nhà** tại TP. Hồ Chí Minh.
- Giữ nhóm nhà ở: `Nhà`, `Biệt thự/Nhà liền kề` và `Nhà mặt phố` (nếu có).
- Loại căn hộ chung cư, đất nền và tin cho thuê.
- Không tự tạo giá ngẫu nhiên hoặc điền nhãn giá giả.

## Nguồn dữ liệu

Tinix AI, **Vietnam Real Estate Listings (2025–2026)** trên Hugging Face:

https://huggingface.co/datasets/tinixai/vietnam-real-estates

Dataset công bố 3.500.744 tin đăng từ 01/06/2025 đến 31/03/2026, bao phủ 63 tỉnh/thành; license CC BY-NC 4.0. Chỉ sử dụng phi thương mại/học tập nếu chưa có thỏa thuận cấp phép khác. Nguồn có cả tin bán và cho thuê, vì vậy dự án lọc theo địa điểm, loại hình và nội dung tin trước khi huấn luyện.

## Cách cập nhật

Chạy tại thư mục gốc dự án:

```bash
pip install -r requirements.txt
python -m backend.update_dataset
```

Lệnh trên dùng streaming, lọc tin TP.HCM và ghi đè `database/dataset.csv` theo cách an toàn (chỉ thay file khi lọc thành công). Muốn thử giới hạn số dòng trước:

```bash
python -m backend.update_dataset --max-rows 10000
```

Sau khi dataset cập nhật xong, huấn luyện mô hình:

```bash
python -m backend.train_model
```

Kết quả được lưu vào `models/trained_model.pkl`, `models/trained_data.csv` và `models/model_report.txt`. Có thể chạy qua trang quản trị như trước; API và giao diện không cần thay đổi.

## Đánh giá

Nếu dataset có `published_at`, chương trình giữ các tin mới nhất làm tập test theo thời gian để mô phỏng việc dự đoán tin đăng tương lai. Nếu không có ngày đăng, chương trình mới dùng chia ngẫu nhiên. Baseline được fit riêng trên tập train để tránh rò rỉ dữ liệu test; sau khi đánh giá, model production được fit lại trên toàn bộ dữ liệu sạch.

Trường `accuracy` vẫn được giữ trong model bundle để tương thích giao diện cũ, nhưng giá trị này là R² trên log giá được quy đổi ra phần trăm, **không phải accuracy của bài toán phân loại**. Báo cáo mới hiển thị rõ MAE, RMSE và R².
