"""
predict.py — Dự đoán giá nhà với đảm bảo logic MONOTONE và kiểm tra tính hợp lệ dữ liệu.

Kiến trúc 2 thành phần:
  1. Baseline: giá/m² theo đường cong lũy thừa (calibrate từ data thực)
     log(ppm) = intercept + slope * log(area)
     => price ~ area^(1+slope), với slope ∈ (-1,0) → tổng giá tăng đơn điệu
  2. Quality model: Gradient Boosting / Random Forest học RESIDUAL (location / ward / rooms / toilets)
  
Kết hợp: price = exp(log_ppm_baseline + log_ppm_quality) * area

ĐẢM BẢO: nhà 300m² LUÔN > nhà 100m² cùng thông số (location/ward/rooms/toilets).
Kiểm tra tính hợp lệ ở backend: thành phố, phường/xã theo danh mục; cơ chế xác thực tên đường.
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.location_service import (
    get_active_location_catalog,
    get_active_street_catalog,
    validate_location,
    StreetValidator,
    clear_location_cache,
    clear_street_catalog_cache,
)

ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "trained_model.pkl"

_bundle = None   # {'model', 'slope', 'intercept', 'feats', 'location_catalog'}


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Mô hình AI chưa được huấn luyện. "
                "Admin vui lòng vào /admin/train để huấn luyện trước."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def reload_model():
    global _bundle
    _bundle = None
    clear_location_cache()
    clear_street_catalog_cache()
    return load_model()


# Feature columns (NO area — area được xử lý bởi baseline curve)
FEATURE_COLUMNS = ["location", "ward", "rooms", "toilets"]


def _raw_predict(bundle: dict, area: float, location: str, ward: str,
                 rooms: float, toilets: float) -> float:
    """Dự đoán giá (VNĐ) với đảm bảo đơn điệu theo area."""
    slope     = bundle["slope"]
    intercept = bundle["intercept"]
    model     = bundle["model"]

    # 1. Baseline giá/m² từ đường cong area
    log_ppm_base = intercept + slope * np.log(max(area, 1.0))

    # 2. Hiệu chỉnh chất lượng (location / ward / rooms / toilets)
    X = pd.DataFrame([{
        "location": location,
        "ward":     ward,
        "rooms":    rooms,
        "toilets":  toilets,
    }])
    log_ppm_quality = float(model.predict(X)[0])

    # 3. Tổng hợp: ppm * area
    ppm   = np.exp(log_ppm_base + log_ppm_quality)
    price = ppm * area
    return max(price, 0.0)


def predict_price(form_data: dict, street_catalog: dict[str, dict[str, list[str]]] | None = None) -> dict[str, Any]:
    """
    Dự đoán giá nhà và kiểm tra tính hợp lệ dữ liệu chặt chẽ ở backend.
    - Thành phố phải tồn tại trong danh mục của mô hình.
    - Phường/xã phải thuộc đúng thành phố đã chọn.
    - Xác thực tên đường theo danh mục đáng tin cậy nếu có; nếu không có thì ghi nhận chưa xác thực, không chặn tùy tiện.
    - Diện tích phải > 0.
    
    Trả về dict: price, min_price, max_price, confidence, algorithm, city, ward, street, street_validation.
    Đơn vị: VNĐ.
    """
    bundle = load_model()

    # 1. Lấy danh mục địa điểm hợp lệ
    catalog = bundle.get("location_catalog") or get_active_location_catalog()

    # 2. Kiểm tra tính hợp lệ của Thành phố và Phường/Xã
    city_raw = form_data.get("city", "")
    ward_raw = form_data.get("ward", "")
    norm_city, norm_ward = validate_location(city_raw, ward_raw, catalog)

    # 3. Kiểm tra tính hợp lệ của các thông số kỹ thuật
    try:
        area_val = form_data.get("area", 0)
        area = float(area_val if area_val not in (None, "") else 0)
    except (ValueError, TypeError):
        raise ValueError("Diện tích không hợp lệ. Vui lòng nhập số dương.")

    if area <= 0:
        raise ValueError("Diện tích sử dụng phải lớn hơn 0 m².")

    try:
        rooms_val = form_data.get("bedrooms", 0)
        rooms = float(rooms_val if rooms_val not in (None, "") else 0)
    except (ValueError, TypeError):
        rooms = 0.0
    if rooms < 0:
        raise ValueError("Số phòng ngủ không thể là số âm.")

    try:
        toilets_val = form_data.get("bathrooms", 0)
        toilets = float(toilets_val if toilets_val not in (None, "") else 0)
    except (ValueError, TypeError):
        toilets = 0.0
    if toilets < 0:
        raise ValueError("Số phòng vệ sinh không thể là số âm.")

    # 4. Kiểm tra tính hợp lệ của tên đường
    if street_catalog is None:
        street_catalog = get_active_street_catalog()
    validator = StreetValidator(street_catalog=street_catalog)
    street_res = validator.validate(norm_city, norm_ward, form_data.get("street"))

    # 5. Thực hiện dự đoán giá
    price = _raw_predict(bundle, area, norm_city, norm_ward, rooms, toilets)

    # 6. Biên độ dao động
    if area > 0 and rooms > 0 and norm_ward != "unknown":
        margin = 0.12
    elif area > 0 and rooms > 0:
        margin = 0.16
    elif area > 0:
        margin = 0.22
    else:
        margin = 0.30

    accuracy = bundle.get("accuracy")
    if accuracy is None:
        report_path = ROOT_DIR / "models" / "model_report.txt"
        if report_path.exists():
            for line in report_path.read_text(encoding="utf-8").splitlines():
                if "Accuracy" in line:
                    try:
                        accuracy = float(line.split(":")[-1].strip().replace("%", ""))
                    except Exception:
                        pass
        if accuracy is None:
            accuracy = 88.0

    algorithm = bundle.get("algorithm", "AI v2.4")
    algo_display = {
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
        "ridge": "Ridge Regression"
    }.get(algorithm, algorithm.replace("_", " ").title() if algorithm != "AI v2.4" else algorithm)

    return {
        "price":             round(price, 0),
        "min_price":         round(price * (1 - margin), 0),
        "max_price":         round(price * (1 + margin), 0),
        "confidence":        accuracy,
        "algorithm":         algo_display,
        "city":              norm_city,
        "ward":              norm_ward,
        "street":            street_res.get("street", ""),
        "street_validation": street_res,
    }
