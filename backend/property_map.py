"""
Module xử lý dữ liệu Bản đồ Bất động sản và Tọa độ Địa lý (Real Estate Map & Geocoding Service)
HousePrice AI Platform
"""
import json
import os
import re
import unicodedata
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd
from backend.location_service import normalize_city, normalize_string

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "database" / "dataset.csv"
GEOCODE_CACHE_PATH = ROOT_DIR / "models" / "geocode_cache.json"

# Tọa độ trung tâm hành chính chính thức của các Tỉnh/Thành phố và Quận/Huyện tại Việt Nam
OFFICIAL_DISTRICT_COORDS = {
    # --- TP. HỒ CHÍ MINH ---
    ("Tp Hồ Chí Minh", "Quận 1"): {"lat": 10.7769, "lng": 106.7009},
    ("Tp Hồ Chí Minh", "Quận 3"): {"lat": 10.7844, "lng": 106.6844},
    ("Tp Hồ Chí Minh", "Quận 4"): {"lat": 10.7578, "lng": 106.7012},
    ("Tp Hồ Chí Minh", "Quận 5"): {"lat": 10.7540, "lng": 106.6634},
    ("Tp Hồ Chí Minh", "Quận 6"): {"lat": 10.7481, "lng": 106.6352},
    ("Tp Hồ Chí Minh", "Quận 7"): {"lat": 10.7340, "lng": 106.7218},
    ("Tp Hồ Chí Minh", "Quận 8"): {"lat": 10.7241, "lng": 106.6286},
    ("Tp Hồ Chí Minh", "Quận 10"): {"lat": 10.7672, "lng": 106.6669},
    ("Tp Hồ Chí Minh", "Quận 11"): {"lat": 10.7629, "lng": 106.6504},
    ("Tp Hồ Chí Minh", "Quận 12"): {"lat": 10.8672, "lng": 106.6413},
    ("Tp Hồ Chí Minh", "Quận Bình Thạnh"): {"lat": 10.8106, "lng": 106.6984},
    ("Tp Hồ Chí Minh", "Quận Bình Tân"): {"lat": 10.7654, "lng": 106.6038},
    ("Tp Hồ Chí Minh", "Quận Gò Vấp"): {"lat": 10.8388, "lng": 106.6653},
    ("Tp Hồ Chí Minh", "Quận Phú Nhuận"): {"lat": 10.7992, "lng": 106.6803},
    ("Tp Hồ Chí Minh", "Quận Tân Bình"): {"lat": 10.8015, "lng": 106.6526},
    ("Tp Hồ Chí Minh", "Quận Tân Phú"): {"lat": 10.7901, "lng": 106.6280},
    ("Tp Hồ Chí Minh", "Thành phố Thủ Đức"): {"lat": 10.8494, "lng": 106.7719},
    ("Tp Hồ Chí Minh", "Quận 2"): {"lat": 10.7872, "lng": 106.7498},
    ("Tp Hồ Chí Minh", "Quận 9"): {"lat": 10.8428, "lng": 106.8286},
    ("Tp Hồ Chí Minh", "Huyện Bình Chánh"): {"lat": 10.6874, "lng": 106.5939},
    ("Tp Hồ Chí Minh", "Huyện Hóc Môn"): {"lat": 10.8841, "lng": 106.5925},
    ("Tp Hồ Chí Minh", "Huyện Củ Chi"): {"lat": 11.0067, "lng": 106.5132},
    ("Tp Hồ Chí Minh", "Huyện Nhà Bè"): {"lat": 10.6953, "lng": 106.7323},
    ("Tp Hồ Chí Minh", "Huyện Cần Giờ"): {"lat": 10.4114, "lng": 106.9546},

    # --- HÀ NỘI ---
    ("Hà Nội", "Quận Ba Đình"): {"lat": 21.0341, "lng": 105.8242},
    ("Hà Nội", "Quận Hoàn Kiếm"): {"lat": 21.0285, "lng": 105.8542},
    ("Hà Nội", "Quận Tây Hồ"): {"lat": 21.0715, "lng": 105.8230},
    ("Hà Nội", "Quận Long Biên"): {"lat": 21.0362, "lng": 105.8927},
    ("Hà Nội", "Quận Cầu Giấy"): {"lat": 21.0313, "lng": 105.7938},
    ("Hà Nội", "Quận Đống Đa"): {"lat": 21.0181, "lng": 105.8299},
    ("Hà Nội", "Quận Hai Bà Trưng"): {"lat": 21.0090, "lng": 105.8569},
    ("Hà Nội", "Quận Hoàng Mai"): {"lat": 20.9765, "lng": 105.8445},
    ("Hà Nội", "Quận Thanh Xuân"): {"lat": 20.9937, "lng": 105.8117},
    ("Hà Nội", "Quận Nam Từ Liêm"): {"lat": 21.0125, "lng": 105.7661},
    ("Hà Nội", "Quận Bắc Từ Liêm"): {"lat": 21.0633, "lng": 105.7626},
    ("Hà Nội", "Quận Hà Đông"): {"lat": 20.9632, "lng": 105.7656},
    ("Hà Nội", "Thị xã Sơn Tây"): {"lat": 21.1353, "lng": 105.5036},
    ("Hà Nội", "Huyện Gia Lâm"): {"lat": 21.0182, "lng": 105.9525},
    ("Hà Nội", "Huyện Đông Anh"): {"lat": 21.1378, "lng": 105.8465},
    ("Hà Nội", "Huyện Sóc Sơn"): {"lat": 21.2825, "lng": 105.8482},
    ("Hà Nội", "Huyện Thanh Trì"): {"lat": 20.9421, "lng": 105.8461},

    # --- ĐÀ NẴNG ---
    ("Đà Nẵng", "Quận Hải Châu"): {"lat": 16.0544, "lng": 108.2208},
    ("Đà Nẵng", "Quận Thanh Khê"): {"lat": 16.0618, "lng": 108.1884},
    ("Đà Nẵng", "Quận Sơn Trà"): {"lat": 16.0825, "lng": 108.2433},
    ("Đà Nẵng", "Quận Ngũ Hành Sơn"): {"lat": 16.0028, "lng": 108.2505},
    ("Đà Nẵng", "Quận Liên Chiểu"): {"lat": 16.0898, "lng": 108.1442},
    ("Đà Nẵng", "Quận Cẩm Lệ"): {"lat": 16.0178, "lng": 108.1994},
    ("Đà Nẵng", "Huyện Hòa Vang"): {"lat": 15.9922, "lng": 108.1250},

    # --- BÌNH DƯƠNG ---
    ("Bình Dương", "Thành phố Thủ Dầu Một"): {"lat": 10.9805, "lng": 106.6519},
    ("Bình Dương", "Thành phố Dĩ An"): {"lat": 10.9069, "lng": 106.7719},
    ("Bình Dương", "Thành phố Thuận An"): {"lat": 10.9234, "lng": 106.7009},
    ("Bình Dương", "Thị xã Bến Cát"): {"lat": 11.1309, "lng": 106.6022},
    ("Bình Dương", "Thành phố Tân Uyên"): {"lat": 11.0825, "lng": 106.7825},

    # --- ĐỒNG NAI ---
    ("Đồng Nai", "Thành phố Biên Hòa"): {"lat": 10.9574, "lng": 106.8427},
    ("Đồng Nai", "Huyện Long Thành"): {"lat": 10.7819, "lng": 106.9582},
    ("Đồng Nai", "Huyện Nhơn Trạch"): {"lat": 10.6622, "lng": 106.8772},
    ("Đồng Nai", "Huyện Trảng Bom"): {"lat": 10.9619, "lng": 107.0125},

    # --- KHÁNH HÒA ---
    ("Khánh Hòa", "Thành phố Nha Trang"): {"lat": 12.2388, "lng": 109.1967},
    # --- BÀ RỊA - VŨNG TÀU ---
    ("Bà Rịa - Vũng Tàu", "Thành phố Vũng Tàu"): {"lat": 10.3460, "lng": 107.0843},
    # --- CẦN THƠ ---
    ("Cần Thơ", "Quận Ninh Kiều"): {"lat": 10.0342, "lng": 105.7797},
    # --- LÂM ĐỒNG ---
    ("Lâm Đồng", "Thành phố Đà Lạt"): {"lat": 11.9404, "lng": 108.4583},
}

# Tọa độ mặc định trung tâm theo Tỉnh/Thành phố
PROVINCE_CENTERS = {
    "Tp Hồ Chí Minh": {"lat": 10.8231, "lng": 106.6297, "zoom": 11},
    "Hà Nội": {"lat": 21.0285, "lng": 105.8542, "zoom": 12},
    "Đà Nẵng": {"lat": 16.0544, "lng": 108.2022, "zoom": 12},
    "Bình Dương": {"lat": 10.9805, "lng": 106.6519, "zoom": 11},
    "Đồng Nai": {"lat": 10.9574, "lng": 106.8427, "zoom": 11},
    "Cần Thơ": {"lat": 10.0342, "lng": 105.7797, "zoom": 12},
    "Bà Rịa - Vũng Tàu": {"lat": 10.3460, "lng": 107.0843, "zoom": 12},
    "Khánh Hòa": {"lat": 12.2388, "lng": 109.1967, "zoom": 12},
    "Lâm Đồng": {"lat": 11.9404, "lng": 108.4583, "zoom": 12},
}

# Phân cấp 5 ngưỡng giá chuẩn theo phân vị thực tế của thị trường
PRICE_TIERS = [
    {"id": "low", "label": "Dưới 2.5 Tỷ", "max": 2_500_000_000, "color": "#10B981", "badge": "Thấp"},
    {"id": "mid_low", "label": "2.5 – 5 Tỷ", "max": 5_000_000_000, "color": "#3B82F6", "badge": "Trung bình thấp"},
    {"id": "mid_high", "label": "5 – 8 Tỷ", "max": 8_000_000_000, "color": "#F59E0B", "badge": "Trung bình cao"},
    {"id": "high", "label": "8 – 15 Tỷ", "max": 15_000_000_000, "color": "#EA580C", "badge": "Cao"},
    {"id": "very_high", "label": "Trên 15 Tỷ", "max": float("inf"), "color": "#EF4444", "badge": "Rất cao"},
]


def load_geocode_cache() -> dict:
    """Tải bộ nhớ cache tọa độ geocoding."""
    if GEOCODE_CACHE_PATH.exists():
        try:
            return json.loads(GEOCODE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_geocode_cache(cache: dict) -> None:
    """Lưu an toàn bộ nhớ cache tọa độ geocoding."""
    GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = GEOCODE_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(GEOCODE_CACHE_PATH)


def get_district_centroid(province: str, district: str) -> dict:
    """Tìm tọa độ tâm hành chính chính thức của Quận/Huyện."""
    canon_p = normalize_city(province) if province else ""
    p_norm = unicodedata.normalize("NFC", str(canon_p or province)).strip().lower()
    d_norm = unicodedata.normalize("NFC", str(district)).strip().lower()

    for (p, d), coords in OFFICIAL_DISTRICT_COORDS.items():
        prov_match = (unicodedata.normalize("NFC", p).strip().lower() == p_norm) or (bool(canon_p) and normalize_city(p) == canon_p)
        if prov_match:
            if unicodedata.normalize("NFC", d).strip().lower() == d_norm:
                return coords

    # Thử tìm tương đối theo tên quận
    for (p, d), coords in OFFICIAL_DISTRICT_COORDS.items():
        prov_match = (unicodedata.normalize("NFC", p).strip().lower() == p_norm) or (bool(canon_p) and normalize_city(p) == canon_p)
        if prov_match:
            d_clean = re.sub(r"^(quận|huyện|thành phố|thị xã|tp\.?|q\.)\s+", "", d_norm)
            k_clean = re.sub(r"^(quận|huyện|thành phố|thị xã|tp\.?|q\.)\s+", "", unicodedata.normalize("NFC", d).strip().lower())
            if d_clean == k_clean or d_clean in k_clean:
                return coords

    # Fallback về tâm Tỉnh/Thành
    if canon_p in PROVINCE_CENTERS:
        c = PROVINCE_CENTERS[canon_p]
        return {"lat": c["lat"], "lng": c["lng"]}
    for p, c in PROVINCE_CENTERS.items():
        if unicodedata.normalize("NFC", p).strip().lower() == p_norm:
            return {"lat": c["lat"], "lng": c["lng"]}

    return {"lat": 10.8231, "lng": 106.6297}


def classify_price_tier(price: float) -> dict:
    """Xếp loại khoảng giá và màu sắc hiển thị."""
    for tier in PRICE_TIERS:
        if price <= tier["max"]:
            return tier
    return PRICE_TIERS[-1]


def get_property_map_data(province: str = "Tp Hồ Chí Minh",
                           district: str = "",
                           min_price: float = 0,
                           max_price: float = 0,
                           limit: int = 500) -> dict:
    """
    Truy vấn dữ liệu bất động sản thực tế và sinh danh sách Marker + Heatmap.
    Không tạo tọa độ ảo ngẫu nhiên; định vị chính xác theo cache geocode hoặc tâm khu vực có nhãn rõ ràng.
    """
    if not DATASET_PATH.exists():
        canon_p = normalize_city(province) if province else "Tp Hồ Chí Minh"
        return {
            "success": False,
            "items": [],
            "heatmap": [],
            "center": PROVINCE_CENTERS.get(canon_p, {"lat": 10.8231, "lng": 106.6297, "zoom": 11}),
            "tiers": PRICE_TIERS,
            "total": 0,
            "displayed": 0
        }

    df = pd.read_csv(DATASET_PATH)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["location"] = df["location"].fillna("").astype(str).str.strip()
    df["district"] = df["district"].fillna("").astype(str).str.strip()
    df["ward"] = df["ward"].fillna("").astype(str).str.strip()
    df["address"] = df["address"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("Bất động sản").astype(str).str.strip()

    # Lọc phân khúc bán hợp lệ
    mask = (df["price"] >= 100_000_000) & (df["area"] >= 10)

    canon_p = normalize_city(province) if province else ""
    if province:
        p_norm = unicodedata.normalize("NFC", str(canon_p or province)).strip().lower()
        def _match_prov(val):
            if not val:
                return False
            val_canon = normalize_city(str(val))
            if canon_p and val_canon == canon_p:
                return True
            return unicodedata.normalize("NFC", str(val)).strip().lower() == p_norm
        mask &= df["location"].apply(_match_prov)

    if district:
        d_norm = unicodedata.normalize("NFC", str(district)).strip().lower()
        d_clean = re.sub(r"^(quận|huyện|thành phố|thị xã|tp\.?|q\.)\s+", "", d_norm)
        def _match_dist(val):
            if not val:
                return False
            v_norm = unicodedata.normalize("NFC", str(val)).strip().lower()
            if v_norm == d_norm:
                return True
            v_clean = re.sub(r"^(quận|huyện|thành phố|thị xã|tp\.?|q\.)\s+", "", v_norm)
            return v_clean == d_clean or (len(d_clean) >= 2 and d_clean in v_clean)
        mask &= df["district"].apply(_match_dist)

    if min_price > 0:
        mask &= (df["price"] >= min_price)
    if max_price > 0:
        mask &= (df["price"] <= max_price)

    filtered = df[mask].copy()
    total_matching = len(filtered)

    # Giới hạn số lượng điểm hiển thị để đảm bảo hiệu năng và không lag trình duyệt
    sample_df = filtered.head(min(limit, 800))
    cache = load_geocode_cache()

    items = []
    heatmap_points = []

    for idx, row in sample_df.iterrows():
        p_id = str(row.get("listing_id") or idx)
        title = row["title"]
        price = float(row["price"])
        area = float(row["area"])
        ppm = price / area if area > 0 else 0
        loc = row["location"]
        dist = row["district"]
        ward = row["ward"]
        addr = row["address"]
        rooms = int(row["rooms"]) if pd.notna(row.get("rooms")) and row.get("rooms") > 0 else 0
        toilets = int(row["toilets"]) if pd.notna(row.get("toilets")) and row.get("toilets") > 0 else 0

        # Kiểm tra xem địa chỉ này đã có tọa độ cụ thể trong cache chưa
        full_addr_key = f"{addr}, {ward}, {dist}, {loc}".strip(", ")
        tier_info = classify_price_tier(price)

        if full_addr_key in cache:
            cached = cache[full_addr_key]
            lat = float(cached["lat"])
            lng = float(cached["lng"])
            is_exact = True
            loc_label = f"{addr}, {ward}" if addr else ward
        else:
            # Dùng tọa độ tâm hành chính của quận với độ lệch vi mô phân tán đều (micro-dispersion)
            centroid = get_district_centroid(loc, dist)
            # Dùng hash id bất động sản để tạo độ phân tán có tính tất định (không ngẫu nhiên lộn xộn)
            offset_seed = abs(hash(p_id)) % 10000
            angle = (offset_seed / 10000.0) * 2 * np.pi
            # Bán kính vi mô tối đa ~400-800m trong quận (1 độ vĩ ~111km -> 0.005 độ ~ 500m)
            radius = ((abs(hash(p_id + "rad")) % 5000) / 5000.0) * 0.006 + 0.001
            lat = centroid["lat"] + radius * np.cos(angle)
            lng = centroid["lng"] + (radius * 1.05) * np.sin(angle)
            is_exact = False
            loc_label = f"Khu vực {dist}, {loc}"

        # Định dạng tiền
        if price >= 1_000_000_000:
            price_str = f"{price / 1_000_000_000:.2f} Tỷ"
        else:
            price_str = f"{price / 1_000_000:.0f} Triệu"

        ppm_str = f"{ppm / 1_000_000:.1f} Tr/m²"

        item = {
            "id": p_id,
            "title": title[:65] + ("..." if len(title) > 65 else ""),
            "price": price,
            "price_formatted": price_str,
            "price_per_m2": round(ppm, 0),
            "price_per_m2_formatted": ppm_str,
            "area": round(area, 1),
            "rooms": rooms,
            "toilets": toilets,
            "location": loc,
            "district": dist,
            "ward": ward,
            "address": addr,
            "location_label": loc_label,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "is_exact": is_exact,
            "tier_id": tier_info["id"],
            "tier_label": tier_info["label"],
            "color": tier_info["color"],
        }
        items.append(item)

        # Trọng số cho lớp Heatmap theo đơn giá trên mỗi m²
        # Đơn vị: ppm / 10 triệu (VD: 150 triệu/m² có trọng số 15)
        weight = float(np.clip(ppm / 10_000_000, 1.0, 35.0))
        heatmap_points.append({
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "weight": round(weight, 2),
        })

    # Xác định tâm bản đồ mặc định
    default_center = {"lat": 10.8231, "lng": 106.6297, "zoom": 11}
    center_info = PROVINCE_CENTERS.get(canon_p, PROVINCE_CENTERS.get(province, default_center))
    if district:
        d_centroid = get_district_centroid(province, district)
        center_info = {"lat": d_centroid["lat"], "lng": d_centroid["lng"], "zoom": 13}

    return {
        "success": True,
        "total": total_matching,
        "displayed": len(items),
        "province": province,
        "district": district,
        "center": center_info,
        "tiers": PRICE_TIERS,
        "items": items,
        "heatmap": heatmap_points,
    }


def geocode_address_google(address: str, api_key: str = None) -> tuple[float, float, bool]:
    """
    Geocode địa chỉ sang Lat/Lng sử dụng Google Geocoding API với bộ nhớ đệm vĩnh viễn.
    Nếu không có key hoặc lỗi mạng, trả về None và không ghi đè dữ liệu.
    """
    if not address or not address.strip():
        return None

    cache = load_geocode_cache()
    norm_addr = address.strip()
    if norm_addr in cache:
        c = cache[norm_addr]
        return c["lat"], c["lng"], True

    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return None

    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json?" + urllib.parse.urlencode({
            "address": norm_addr,
            "key": api_key,
            "language": "vi",
            "region": "vn",
        })
        req = urllib.request.Request(url, headers={"User-Agent": "HousePriceAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                lat, lng = float(loc["lat"]), float(loc["lng"])
                cache[norm_addr] = {
                    "lat": lat,
                    "lng": lng,
                    "formatted_address": data["results"][0].get("formatted_address", ""),
                    "status": "OK"
                }
                save_geocode_cache(cache)
                return lat, lng, True
    except Exception as e:
        print(f"[WARN] Geocoding API error: {e}")

    return None
