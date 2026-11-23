"""
Blueprint định tuyến cho Chức năng Bản đồ giá bất động sản Leaflet & OpenStreetMap
HousePrice AI Platform
"""
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify
from backend.comparison import get_available_provinces, get_districts_by_province
from backend.property_map import get_property_map_data

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
map_bp = Blueprint("map_views", __name__)


@map_bp.route("/map")
def map_page():
    """Trang giao diện bản đồ giá bất động sản trực tiếp với Leaflet & OpenStreetMap."""
    provinces = get_available_provinces()
    default_province = "Tp Hồ Chí Minh" if "Tp Hồ Chí Minh" in provinces else (provinces[0] if provinces else "")
    default_districts = get_districts_by_province(default_province) if default_province else []

    try:
        initial_data = get_property_map_data(default_province, limit=450)
    except Exception as e:
        print(f"[WARN] Error fetching initial map data: {e}")
        initial_data = {
            "success": True,
            "items": [],
            "heatmap": [],
            "center": {"lat": 10.8231, "lng": 106.6297, "zoom": 11},
            "total": 0,
            "displayed": 0,
            "province": default_province,
            "district": "",
        }

    return render_template(
        "map.html",
        provinces=provinces,
        default_province=default_province,
        districts=default_districts,
        initial_data=initial_data,
    )


@map_bp.route("/api/map/config")
def api_map_config():
    """Thông tin cấu hình bản đồ Leaflet OpenStreetMap (hoàn toàn miễn phí, không cần API Key)."""
    return jsonify({
        "success": True,
        "provider": "leaflet_osm",
        "default_center": {"lat": 10.8231, "lng": 106.6297, "zoom": 11},
        "default_province": "Tp Hồ Chí Minh",
        "attribution": "© OpenStreetMap contributors"
    })


@map_bp.route("/api/properties/map")
def api_property_map():
    """API lấy dữ liệu Marker bất động sản và tọa độ Heatmap."""
    province = request.args.get("province", "Tp Hồ Chí Minh").strip()
    district = request.args.get("district", "").strip()
    min_price = float(request.args.get("min_price", 0) or 0)
    max_price = float(request.args.get("max_price", 0) or 0)
    limit = int(request.args.get("limit", 400) or 400)

    try:
        data = get_property_map_data(
            province=province,
            district=district,
            min_price=min_price,
            max_price=max_price,
            limit=limit,
        )
        return jsonify(data)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Lỗi truy vấn dữ liệu bản đồ: {e}",
            "items": [],
            "heatmap": []
        }), 500


@map_bp.route("/api/properties/geo-stats")
def api_geo_stats():
    """API thống kê mật độ và phân vị giá khu vực trên bản đồ."""
    province = request.args.get("province", "Tp Hồ Chí Minh").strip()
    district = request.args.get("district", "").strip()
    data = get_property_map_data(province=province, district=district, limit=600)
    
    total = data.get("total", 0)
    items = data.get("items", [])
    
    avg_price = 0
    avg_ppm = 0
    if items:
        avg_price = sum(item["price"] for item in items) / len(items)
        avg_ppm = sum(item["price_per_m2"] for item in items) / len(items)

    return jsonify({
        "success": True,
        "province": province,
        "district": district,
        "total_properties": total,
        "sample_size": len(items),
        "avg_price": round(avg_price, 0),
        "avg_price_per_m2": round(avg_ppm, 0),
        "tiers": data.get("tiers", []),
    })
