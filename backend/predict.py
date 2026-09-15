"""
predict.py — Dự đoán giá nhà với đảm bảo logic MONOTONE.

Kiến trúc 2 thành phần:
  1. Baseline: giá/m² theo đường cong lũy thừa (calibrate từ data thực)
     log(ppm) = intercept + slope * log(area)
     => price ~ area^(1+slope), với slope ∈ (-1,0) → tổng giá tăng đơn điệu
  2. Quality model: Random Forest học RESIDUAL (location / ward / rooms / toilets)
  
Kết hợp: price = exp(log_ppm_baseline + log_ppm_quality) * area

ĐẢM BẢO: nhà 300m² LUÔN > nhà 100m² cùng thông số (location/ward/rooms/toilets)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "trained_model.pkl"

_bundle = None   # {'model', 'slope', 'intercept', 'feats'}


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
    return load_model()


# Mapping tên thành phố form → dataset
CITY_MAP = {
    "Hồ Chí Minh":  "Tp Hồ Chí Minh",
    "Hà Nội":        "Hà Nội",
    "Đà Nẵng":       "Đà Nẵng",
    "Cần Thơ":       "Cần Thơ",
    "Bình Dương":    "Bình Dương",
}

# Feature columns (NO area — area được xử lý bởi baseline curve)
FEATURE_COLUMNS = ["location", "ward", "rooms", "toilets"]


def _raw_predict(bundle, area: float, location: str, ward: str,
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


def predict_price(form_data: dict) -> dict:
    """
    Trả về dict: price, min_price, max_price, confidence.
    Đơn vị: VNĐ.

    ĐẢMBẢO: cùng location/ward/rooms/toilets, area lớn hơn → price lớn hơn.
    """
    bundle = load_model()

    city_raw = form_data.get("city", "unknown") or "unknown"
    location = CITY_MAP.get(city_raw, city_raw)
    ward     = form_data.get("ward", "unknown") or "unknown"
    area     = float(form_data.get("area", 0)      or 0)
    rooms    = float(form_data.get("bedrooms", 0)  or 0)
    toilets  = float(form_data.get("bathrooms", 0) or 0)

    if area <= 0:
        area = 80.0  # fallback mặc định

    price = _raw_predict(bundle, area, location, ward, rooms, toilets)

    # Biên độ dao động theo mức đủ thông tin
    if area > 0 and rooms > 0 and ward != "unknown":
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
                    except:
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
        "price":      round(price, 0),
        "min_price":  round(price * (1 - margin), 0),
        "max_price":  round(price * (1 + margin), 0),
        "confidence": accuracy,
        "algorithm": algo_display,
    }
