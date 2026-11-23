"""
Module so sánh giá bất động sản giữa các khu vực (Area Comparison Service)
HousePrice AI Platform
"""
from pathlib import Path
import unicodedata
import re
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "database" / "dataset.csv"

# Bảng màu đại diện phân biệt cho tối đa 5 khu vực so sánh
AREA_COLORS = [
    {"primary": "#3B82F6", "sub": "rgba(59, 130, 246, 0.2)", "border": "#60A5FA", "name": "Xanh dương"},
    {"primary": "#8B5CF6", "sub": "rgba(139, 92, 246, 0.2)", "border": "#A78BFA", "name": "Tím"},
    {"primary": "#06B6D4", "sub": "rgba(6, 182, 212, 0.2)", "border": "#22D3EE", "name": "Cyan"},
    {"primary": "#F59E0B", "sub": "rgba(245, 158, 11, 0.2)", "border": "#FBBF24", "name": "Cam"},
    {"primary": "#10B981", "sub": "rgba(16, 185, 129, 0.2)", "border": "#34D399", "name": "Xanh lá"},
]

_DATASET_CACHE = None
_CACHE_MTIME = 0


def _get_raw_dataset() -> pd.DataFrame:
    """Tải dataset với cơ chế cache theo file modified time."""
    global _DATASET_CACHE, _CACHE_MTIME
    if not DATASET_PATH.exists():
        return pd.DataFrame()

    mtime = DATASET_PATH.stat().st_mtime
    if _DATASET_CACHE is None or mtime != _CACHE_MTIME:
        df = pd.read_csv(DATASET_PATH)
        # Chuẩn hóa cột giá và diện tích
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["area"] = pd.to_numeric(df["area"], errors="coerce")
        df["location"] = df["location"].fillna("").astype(str).str.strip()
        df["district"] = df["district"].fillna("").astype(str).str.strip()
        _DATASET_CACHE = df
        _CACHE_MTIME = mtime
    return _DATASET_CACHE


def format_currency_vnd(price: float) -> str:
    """Định dạng số tiền sang chuẩn Tỷ / Triệu VNĐ."""
    if not np.isfinite(price) or price <= 0:
        return "0 VNĐ"
    if price >= 1_000_000_000:
        return f"{price / 1_000_000_000:.2f} Tỷ"
    elif price >= 1_000_000:
        return f"{price / 1_000_000:.1f} Triệu"
    elif price >= 1_000:
        return f"{price / 1_000:,.0f} Nghìn"
    return f"{price:,.0f} VNĐ"


def format_ppm_vnd(ppm: float) -> str:
    """Định dạng đơn giá trên mỗi m²."""
    if not np.isfinite(ppm) or ppm <= 0:
        return "0 đ/m²"
    if ppm >= 1_000_000_000:
        return f"{ppm / 1_000_000_000:.2f} Tỷ/m²"
    elif ppm >= 1_000_000:
        return f"{ppm / 1_000_000:.1f} Tr/m²"
    return f"{ppm:,.0f} đ/m²"


def get_available_provinces() -> list[str]:
    """Lấy danh sách các tỉnh/thành phố có dữ liệu thực tế."""
    df = _get_raw_dataset()
    if df.empty:
        return ["Tp Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Bình Dương"]

    # Đếm số tin theo tỉnh và sắp xếp giảm dần
    counts = df[df["location"] != ""]["location"].value_counts()
    return counts.index.tolist()


def get_districts_by_province(province: str) -> list[str]:
    """Lấy danh sách các quận/huyện thuộc tỉnh/thành phố chỉ định."""
    df = _get_raw_dataset()
    if df.empty:
        return []

    p_norm = unicodedata.normalize("NFC", province).strip().lower()
    loc_mask = df["location"].apply(lambda x: unicodedata.normalize("NFC", str(x)).strip().lower() == p_norm)
    sub = df[loc_mask & (df["district"] != "")]
    counts = sub["district"].value_counts()
    return counts.index.tolist()


def compare_areas(province: str, districts: list[str],
                  time_range: str = "all", property_type: str = "all") -> dict:
    """
    Phân tích so sánh giá bất động sản giữa các quận/huyện được chọn.
    Trả về dữ liệu thẻ thống kê, biểu đồ Chart.js, bảng so sánh và nhận xét tự động.
    """
    if not districts:
        raise ValueError("Vui lòng chọn ít nhất 1 khu vực để so sánh.")
    if len(districts) > 5:
        raise ValueError("Chỉ được so sánh tối đa 5 khu vực cùng lúc để đảm bảo tính trực quan.")

    df = _get_raw_dataset()
    if df.empty:
        raise ValueError("Không tìm thấy cơ sở dữ liệu bất động sản.")

    p_norm = unicodedata.normalize("NFC", province).strip().lower()
    df_prov = df[df["location"].apply(lambda x: unicodedata.normalize("NFC", str(x)).strip().lower() == p_norm)].copy()

    # Lọc phân khúc bán sạch (giá >= 100M, diện tích >= 10m²)
    clean_mask = (df_prov["price"] >= 100_000_000) & (df_prov["area"] >= 10)
    df_clean = df_prov[clean_mask].copy()
    df_clean["price_per_m2"] = df_clean["price"] / df_clean["area"]

    areas_results = []
    for idx, d_name in enumerate(districts):
        d_norm = unicodedata.normalize("NFC", d_name).strip().lower()
        sub = df_clean[df_clean["district"].apply(lambda x: unicodedata.normalize("NFC", str(x)).strip().lower() == d_norm)]
        
        n_samples = len(sub)
        color_scheme = AREA_COLORS[idx % len(AREA_COLORS)]

        if n_samples == 0:
            areas_results.append({
                "name": d_name,
                "province": province,
                "color": color_scheme["primary"],
                "color_sub": color_scheme["sub"],
                "color_border": color_scheme["border"],
                "count": 0,
                "avg_price": 0,
                "median_price": 0,
                "min_price": 0,
                "max_price": 0,
                "avg_price_per_m2": 0,
                "median_price_per_m2": 0,
                "avg_area": 0,
                "price_formatted": "Không có dữ liệu",
                "median_formatted": "—",
                "min_formatted": "—",
                "max_formatted": "—",
                "ppm_formatted": "—",
                "area_formatted": "—",
                "sample_warning": "Khu vực này hiện chưa có dữ liệu giao dịch phù hợp.",
                "price_q25": 0,
                "price_q75": 0,
            })
            continue

        prices = sub["price"].to_numpy()
        ppms = sub["price_per_m2"].to_numpy()
        areas = sub["area"].to_numpy()

        avg_p = float(np.mean(prices))
        med_p = float(np.median(prices))
        min_p = float(np.min(prices))
        max_p = float(np.max(prices))
        q25_p = float(np.percentile(prices, 25))
        q75_p = float(np.percentile(prices, 75))

        avg_ppm = float(np.mean(ppms))
        med_ppm = float(np.median(ppms))
        avg_a = float(np.mean(areas))

        warning = ""
        if n_samples < 15:
            warning = f"Số lượng mẫu hạn chế ({n_samples} tin). Số liệu mang tính tham khảo."

        areas_results.append({
            "name": d_name,
            "province": province,
            "color": color_scheme["primary"],
            "color_sub": color_scheme["sub"],
            "color_border": color_scheme["border"],
            "count": int(n_samples),
            "avg_price": avg_p,
            "median_price": med_p,
            "min_price": min_p,
            "max_price": max_p,
            "avg_price_per_m2": avg_ppm,
            "median_price_per_m2": med_ppm,
            "avg_area": avg_a,
            "price_formatted": format_currency_vnd(avg_p),
            "median_formatted": format_currency_vnd(med_p),
            "min_formatted": format_currency_vnd(min_p),
            "max_formatted": format_currency_vnd(max_p),
            "ppm_formatted": format_ppm_vnd(avg_ppm),
            "area_formatted": f"{avg_a:.1f} m²",
            "sample_warning": warning,
            "price_q25": q25_p,
            "price_q75": q75_p,
        })

    # Xây dựng các cấu trúc biểu đồ Chart.js
    labels = [a["name"] for a in areas_results]
    colors = [a["color"] for a in areas_results]
    borders = [a["color_border"] for a in areas_results]

    chart_avg_price = {
        "labels": labels,
        "datasets": [{
            "label": "Giá trung bình (Tỷ VNĐ)",
            "data": [round(a["avg_price"] / 1_000_000_000, 2) for a in areas_results],
            "backgroundColor": colors,
            "borderColor": borders,
            "borderWidth": 1.5,
            "borderRadius": 8,
        }]
    }

    chart_ppm = {
        "labels": labels,
        "datasets": [{
            "label": "Đơn giá trung bình (Triệu VNĐ/m²)",
            "data": [round(a["avg_price_per_m2"] / 1_000_000, 1) for a in areas_results],
            "backgroundColor": colors,
            "borderColor": borders,
            "borderWidth": 1.5,
            "borderRadius": 8,
        }]
    }

    chart_price_range = {
        "labels": labels,
        "datasets": [
            {
                "label": "Mức thấp (Q25)",
                "data": [round(a["price_q25"] / 1_000_000_000, 2) for a in areas_results],
                "backgroundColor": "rgba(16, 185, 129, 0.7)",
                "borderRadius": 6,
            },
            {
                "label": "Trung vị (Median)",
                "data": [round(a["median_price"] / 1_000_000_000, 2) for a in areas_results],
                "backgroundColor": "rgba(59, 130, 246, 0.7)",
                "borderRadius": 6,
            },
            {
                "label": "Mức cao (Q75)",
                "data": [round(a["price_q75"] / 1_000_000_000, 2) for a in areas_results],
                "backgroundColor": "rgba(236, 72, 153, 0.7)",
                "borderRadius": 6,
            }
        ]
    }

    chart_samples = {
        "labels": labels,
        "datasets": [{
            "label": "Số lượng tin đăng",
            "data": [a["count"] for a in areas_results],
            "backgroundColor": colors,
            "borderColor": borders,
            "borderWidth": 1.5,
            "borderRadius": 8,
        }]
    }

    # Sinh nhận xét tự động dựa trên số liệu thực tế
    insights = _generate_comparison_insights(areas_results, province)

    return {
        "success": True,
        "province": province,
        "areas": areas_results,
        "charts": {
            "avg_price": chart_avg_price,
            "price_per_m2": chart_ppm,
            "price_range": chart_price_range,
            "samples": chart_samples,
        },
        "insights": insights
    }


def _generate_comparison_insights(areas: list[dict], province: str) -> list[dict]:
    """Tạo các phân tích so sánh định lượng hoàn toàn từ số liệu thực tế."""
    valid_areas = [a for a in areas if a["count"] > 0]
    if not valid_areas:
        return [{
            "type": "warn",
            "icon": "ph-warning-circle",
            "text": f"Không tìm thấy mẫu tin giao dịch hợp lệ nào cho các khu vực đã chọn tại {province}."
        }]

    insights = []

    # 1. So sánh giá nhà trung bình
    sorted_by_price = sorted(valid_areas, key=lambda x: x["avg_price"], reverse=True)
    highest_p = sorted_by_price[0]
    lowest_p = sorted_by_price[-1]

    if len(valid_areas) >= 2 and lowest_p["avg_price"] > 0:
        ratio = (highest_p["avg_price"] - lowest_p["avg_price"]) / lowest_p["avg_price"] * 100
        insights.append({
            "type": "price",
            "icon": "ph-trend-up",
            "text": (
                f"Khu vực có giá nhà trung bình cao nhất là <strong>{highest_p['name']}</strong> "
                f"({highest_p['price_formatted']}), cao hơn khoảng <strong>{ratio:.1f}%</strong> "
                f"so với khu vực có giá thấp nhất là <strong>{lowest_p['name']}</strong> ({lowest_p['price_formatted']})."
            )
        })
    else:
        insights.append({
            "type": "price",
            "icon": "ph-tag",
            "text": f"Khu vực <strong>{highest_p['name']}</strong> có mức giá nhà trung bình ghi nhận là <strong>{highest_p['price_formatted']}</strong>."
        })

    # 2. So sánh đơn vị giá trên m²
    sorted_by_ppm = sorted(valid_areas, key=lambda x: x["avg_price_per_m2"], reverse=True)
    highest_ppm = sorted_by_ppm[0]
    lowest_ppm = sorted_by_ppm[-1]

    if len(valid_areas) >= 2 and lowest_ppm["avg_price_per_m2"] > 0:
        ppm_diff = highest_ppm["avg_price_per_m2"] - lowest_ppm["avg_price_per_m2"]
        insights.append({
            "type": "ppm",
            "icon": "ph-ruler",
            "text": (
                f"Về đơn giá theo diện tích đất, <strong>{highest_ppm['name']}</strong> dẫn đầu với "
                f"<strong>{highest_ppm['ppm_formatted']}</strong>, chênh lệch <strong>{format_ppm_vnd(ppm_diff)}</strong> "
                f"so với <strong>{lowest_ppm['name']}</strong> ({lowest_ppm['ppm_formatted']})."
            )
        })

    # 3. So sánh nguồn cung / tính đại diện của dữ liệu
    sorted_by_count = sorted(valid_areas, key=lambda x: x["count"], reverse=True)
    top_supply = sorted_by_count[0]
    insights.append({
        "type": "supply",
        "icon": "ph-database",
        "text": (
            f"Khu vực có mật độ giao dịch và số lượng mẫu phong phú nhất là <strong>{top_supply['name']}</strong> "
            f"với <strong>{top_supply['count']:,}</strong> bản ghi, thể hiện tính đại diện thị trường cao."
        )
    })

    # 4. Cảnh báo mẫu dữ liệu ít
    low_sample_areas = [a["name"] for a in valid_areas if a["count"] < 15]
    if low_sample_areas:
        insights.append({
            "type": "warn",
            "icon": "ph-warning",
            "text": (
                f"Lưu ý: Khu vực <strong>{', '.join(low_sample_areas)}</strong> có số lượng mẫu dưới 15 tin đăng. "
                f"Các chỉ số giá trị có thể chịu ảnh hưởng cục bộ bởi một số bất động sản cá biệt."
            )
        })

    return insights
