    HouseAI — Dự đoán giá nhà bằng AI

Ứng dụng web dự đoán giá bất động sản sử dụng Machine Learning (Gradient Boosting), Flask và MySQL.

**---**

Cấu trúc dự án

\`\`\`
HousePriceAI_Final/
├── app.py                         ← Chạy ứng dụng Flask
├── requirements.txt               ← Các thư viện cần cài
├── README.md
│
├── backend/
│   ├── connect_db.py              ← Kết nối MySQL, khởi tạo bảng
│   ├── auth.py                    ← Đăng ký / Đăng nhập (SHA-256)
│   ├── predict.py                 ← Gọi model AI dự đoán
│   ├── train_model.py             ← Huấn luyện model ML
│   ├── location_service.py        ← Quản lý danh mục địa điểm, chuẩn hóa & xác thực đường
│   └── data_service.py            ← CRUD nhà, lịch sử, thống kê
│
├── frontend/
│   ├── templates/                 ← Giao diện Jinja2 (HTML)
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
│   ├── trained_model.pkl          ← Model đã train (tự tạo)
│   ├── model_report.txt           ← Báo cáo kết quả train
│   ├── location_catalog.json      ← Danh mục địa điểm (tự động tạo sau train)
│   └── street_catalog.json        ← Danh mục đường theo phường (tùy chọn)
│
└── database/
    ├── dataset.csv                ← Dữ liệu huấn luyện (~2400 nhà HCM)
    └── schema.sql                 ← SQL tạo database
\`\`\`

**---**

Yêu cầu hệ thống

- Python 3.10+
- MySQL 8.0+ (hoặc MariaDB 10.5+)
- PyCharm (Community hoặc Professional)

**---**

Hướng dẫn cài đặt & chạy

Bước 1 — Cài thư viện Python

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Bước 2 — Tạo database MySQL (không cần quan tâm)

Mở MySQL Workbench hoặc terminal, chạy:

\`\`\`bash
mysql -u root -p < database/schema.sql
\`\`\`

Hoặc mở file \`database/schema.sql\` và chạy thủ công trong Workbench.

> **Mặc định:** host=localhost, user=root, password="" (không có mật khẩu).
> Nếu MySQL của bạn có mật khẩu, sửa \`backend/connect_db.py\` → \`DB_CONFIG\`.

### Bước 3 — Huấn luyện model AI

\`\`\`bash
python -m backend.train_model
\`\`\`

Kết quả sẽ lưu vào \`models/trained_model.pkl\`.

Để so sánh Linear Regression, Random Forest và Gradient Boosting trên cùng tập train/test mà không thay model đang dùng, chạy:

\`\`\`powershell
py -m backend.train_model --compare-only
\`\`\`

Kết quả MAE, RMSE và R² được in ra màn hình và lưu tại \`models/model_comparison.csv\`.

### Bước 4 — Chạy ứng dụng

\`\`\`bash
python app.py
\`\`\`

Mở trình duyệt: **http://localhost:5000**

**---**

Tài khoản mặc định

| Vai trò       | Tên đăng nhập | Mật khẩu   |
|---------------|--------------|------------|
| Quản trị viên | \`admin\`      | \`admin123\` |
| Người dùng    | (tự đăng ký) | —          |

Đăng nhập Admin: Tích vào **"Đăng nhập với tư cách Quản trị viên"** trên trang login.

**---**

Tính năng

### Người dùng (User)
- Đăng ký / Đăng nhập / Đăng xuất
- Dự đoán giá nhà (chọn thành phố, phường/xã động từ backend, nhập tên đường tùy chọn, diện tích, số phòng...)
- Xem kết quả: giá ước tính, khoảng giá, độ tin cậy, trạng thái xác thực tên đường
- Xem lịch sử dự đoán, xóa từng mục

### Quản trị viên (Admin)
- Dashboard: thống kê tổng quan, lượt dự đoán gần đây
- Quản lý dữ liệu nhà: Thêm / Sửa / Xóa / Tìm kiếm
- Huấn luyện model AI: chọn thuật toán, thông số, tự động cập nhật danh mục địa điểm đồng bộ

**---**

Model AI

- **Thuật toán mặc định:** Gradient Boosting (scikit-learn)
- **Features:** Diện tích, Thành phố, Quận/Phường, Số phòng ngủ, Số WC (không đưa trường đường thiếu cấu trúc vào train)
- **Dataset:** ~2100 bản ghi giá bán nhà tại TP.HCM (sau khi làm sạch & loại ngoại lai)
- **Kết quả:** R² (log) ≈ 0.53, MAE ≈ 1.9 tỷ VNĐ

---

Cấu hình MySQL

Sửa trong \`backend/connect_db.py\`:

\`\`\`python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "",          # ← Đổi nếu có mật khẩu
    "database": "house_prediction",
    "charset":  "utf8mb4",
}
\`\`\`

---

## Cấu hình Danh mục Đường — Phường (\`models/street_catalog.json\`)

### 1. Thực trạng dữ liệu trong dataset hiện có
- Khảo sát thực tế tập dữ liệu \`database/dataset.csv\` cho thấy trường \`address\` bị khuyết thiếu đến **99,63%** (chỉ có 8/2.163 dòng có chuỗi địa chỉ tự do).
- Do đó, **tập dữ liệu huấn luyện hiện tại hoàn toàn không đủ để tự động tạo quan hệ đường → phường**.
- Dự án tuân thủ nguyên tắc: **tuyệt đối không bật xác thực bằng dữ liệu suy đoán từ tiêu đề quảng cáo (\`title\`) hay từ vài dòng dữ liệu thiếu chuẩn hóa**.

### 2. Định dạng file cấu hình danh mục đường
Để kích hoạt tính năng xác thực tên đường theo phường, quản trị viên cần cung cấp danh mục đường có cấu trúc từ nguồn dữ liệu hành chính đáng tin cậy tại đường dẫn \`models/street_catalog.json\` theo định dạng JSON chuẩn:

\`\`\`json
{
  "Tp Hồ Chí Minh": {
    "Phường Bến Nghé": [
      "Đường Đồng Khởi",
      "Đường Nguyễn Huệ"
    ],
    "Phường 1": [
      "Đường Trần Hưng Đạo"
    ]
  }
}
\`\`\`

### 3. Quy tắc hoạt động của hệ thống
- **Chưa cấu hình hoặc file rỗng**: Hệ thống không chặn dự đoán tùy tiện, ghi nhận tên đường và hiển thị trạng thái *"Chưa đối chiếu danh mục đường"*.
- **Lỗi cú pháp file JSON**: Hệ thống hiển thị thông báo lỗi đọc file rõ ràng trong khu vực *"Không thể phân tích"*, không âm thầm coi như danh mục hợp lệ.
- **Đã cấu hình danh mục hợp lệ**:
  - **Đường thuộc đúng phường đã chọn**: Dự đoán tiếp tục thành công, hiển thị huy hiệu *"Đã xác thực thuộc phường"*.
  - **Đường thuộc phường khác**: Từ chối dự đoán, báo rõ đường thuộc phường nào và không thuộc phường đã chọn, giữ nguyên dữ liệu người dùng nhập trên form và không lưu lịch sử cho lần bị từ chối.
  - **Đường không có trong danh mục của phường**: Từ chối dự đoán và thông báo tên đường không có trong danh mục của phường đã chọn.
  - **Bỏ trống tên đường**: Cho phép dự đoán tiếp tục bình thường (tên đường là trường không bắt buộc).


---

## Cấu hình Danh mục Đường — Phường (\`models/street_catalog.json\`)

### 1. Thực trạng dữ liệu trong dataset hiện có
- Khảo sát thực tế tập dữ liệu \`database/dataset.csv\` cho thấy trường \`address\` bị khuyết thiếu đến **99,63%** (chỉ có 8/2.163 dòng có chuỗi địa chỉ tự do).
- Do đó, **tập dữ liệu huấn luyện hiện tại hoàn toàn không đủ để tự động tạo quan hệ đường → phường**.
- Dự án tuân thủ nguyên tắc: **tuyệt đối không bật xác thực bằng dữ liệu suy đoán từ tiêu đề quảng cáo (\`title\`) hay từ vài dòng dữ liệu thiếu chuẩn hóa**.

### 2. Định dạng file cấu hình danh mục đường
Để kích hoạt tính năng xác thực tên đường theo phường, quản trị viên cần cung cấp danh mục đường có cấu trúc từ nguồn dữ liệu hành chính đáng tin cậy tại đường dẫn \`models/street_catalog.json\` theo định dạng JSON chuẩn:

\`\`\`json
{
  "Tp Hồ Chí Minh": {
    "Phường Bến Nghé": [
      "Đường Đồng Khởi",
      "Đường Nguyễn Huệ"
    ],
    "Phường 1": [
      "Đường Trần Hưng Đạo"
    ]
  }
}
\`\`\`

### 3. Quy tắc hoạt động của hệ thống
- **Chưa cấu hình hoặc file rỗng**: Hệ thống không chặn dự đoán tùy tiện, ghi nhận tên đường và hiển thị trạng thái *"Chưa đối chiếu danh mục đường"*.
- **Lỗi cú pháp file JSON**: Hệ thống hiển thị thông báo lỗi đọc file rõ ràng trong khu vực *"Không thể phân tích"*, không âm thầm coi như danh mục hợp lệ.
- **Đã cấu hình danh mục hợp lệ**:
  - **Đường thuộc đúng phường đã chọn**: Dự đoán tiếp tục thành công, hiển thị huy hiệu *"Đã xác thực thuộc phường"*.
  - **Đường thuộc phường khác**: Từ chối dự đoán, báo rõ đường thuộc phường nào và không thuộc phường đã chọn, giữ nguyên dữ liệu người dùng nhập trên form và không lưu lịch sử cho lần bị từ chối.
  - **Đường không có trong danh mục của phường**: Từ chối dự đoán và thông báo tên đường không có trong danh mục của phường đã chọn.
  - **Bỏ trống tên đường**: Cho phép dự đoán tiếp tục bình thường (tên đường là trường không bắt buộc).
\`\`\`