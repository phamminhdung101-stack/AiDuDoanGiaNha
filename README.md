# 🏠 HouseAI — Dự đoán giá nhà bằng AI

Ứng dụng web dự đoán giá bất động sản sử dụng Machine Learning (Gradient Boosting), Flask và MySQL.

---

## 📁 Cấu trúc dự án

```
HousePriceAI_Final/
├── app.py                        ← Chạy ứng dụng Flask
├── requirements.txt              ← Các thư viện cần cài
├── README.md
│
├── backend/
│   ├── connect_db.py             ← Kết nối MySQL, khởi tạo bảng
│   ├── auth.py                   ← Đăng ký / Đăng nhập (SHA-256)
│   ├── predict.py                ← Gọi model AI dự đoán
│   ├── train_model.py            ← Huấn luyện model ML
│   └── data_service.py           ← CRUD nhà, lịch sử, thống kê
│
├── frontend/
│   ├── templates/                ← Giao diện Jinja2 (HTML)
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── predict.html
│   │   ├── history.html
│   │   └── admin/
│   │       ├── dashboard.html
│   │       ├── houses.html
│   │       ├── house_form.html
│   │       └── train.html
│   └── static/
│       ├── css/main.css
│       └── js/main.js
│
├── models/
│   ├── trained_model.pkl         ← Model đã train (tự tạo)
│   └── model_report.txt          ← Báo cáo kết quả train
│
└── database/
    ├── dataset.csv               ← Dữ liệu huấn luyện (~2400 nhà HCM)
    └── schema.sql                ← SQL tạo database
```

---

## ⚙️ Yêu cầu hệ thống

- Python 3.10+
- MySQL 8.0+ (hoặc MariaDB 10.5+)
- PyCharm (Community hoặc Professional)

---

## 🚀 Hướng dẫn cài đặt & chạy

### Bước 1 — Cài thư viện Python

```bash
pip install -r requirements.txt
```

### Bước 2 — Tạo database MySQL (không cần quan tâm)

Mở MySQL Workbench hoặc terminal, chạy:

```bash
mysql -u root -p < database/schema.sql
```

Hoặc mở file `database/schema.sql` và chạy thủ công trong Workbench.

> **Mặc định:** host=localhost, user=root, password="" (không có mật khẩu).
> Nếu MySQL của bạn có mật khẩu, sửa `backend/connect_db.py` → `DB_CONFIG`.

### Bước 3 — Huấn luyện model AI

```bash
python backend/train_model.py
```

Kết quả sẽ lưu vào `models/trained_model.pkl`.

### Bước 4 — Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt: **http://localhost:5000**

---

## 👤 Tài khoản mặc định

| Vai trò       | Tên đăng nhập | Mật khẩu   |
|---------------|--------------|------------|
| Quản trị viên | `admin`      | `admin123` |
| Người dùng    | (tự đăng ký) | —          |

Đăng nhập Admin: Tích vào **"Đăng nhập với tư cách Quản trị viên"** trên trang login.

---

## 📊 Tính năng

### Người dùng (User)
- ✅ Đăng ký / Đăng nhập / Đăng xuất
- ✅ Dự đoán giá nhà (nhập thành phố, quận, diện tích, số phòng...)
- ✅ Xem kết quả: giá ước tính, khoảng giá, độ tin cậy
- ✅ Xem lịch sử dự đoán, xóa từng mục

### Quản trị viên (Admin)
- ✅ Dashboard: thống kê tổng quan, lượt dự đoán gần đây
- ✅ Quản lý dữ liệu nhà: Thêm / Sửa / Xóa / Tìm kiếm
- ✅ Huấn luyện model AI: chọn thuật toán, thông số, xem kết quả R²

---

## 🤖 Model AI

- **Thuật toán mặc định:** Gradient Boosting (scikit-learn)
- **Features:** Diện tích, Thành phố, Quận/Phường, Số phòng ngủ, Số WC
- **Dataset:** ~1001 bản ghi giá bán nhà tại TP.HCM (sau khi lọc)
- **Kết quả:** R² ≈ 0.45, MAE ≈ 1.8 tỷ VNĐ

> Để cải thiện độ chính xác: thu thập thêm dữ liệu (đặc biệt các tỉnh khác, giá/m², pháp lý) và thêm feature mới.

---

## 🛠️ Cấu hình MySQL

Sửa trong `backend/connect_db.py`:

```python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "",          # ← Đổi nếu có mật khẩu
    "database": "house_prediction",
    "charset":  "utf8mb4",
}
```
