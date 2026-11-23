"""
Blueprint định tuyến cho Chức năng So sánh giá bất động sản giữa các khu vực
HousePrice AI Platform
"""
from flask import Blueprint, render_template, request, jsonify
from backend.comparison import (
    get_available_provinces,
    get_districts_by_province,
    compare_areas
)

comparison_bp = Blueprint("comparison_views", __name__)


@comparison_bp.route("/compare")
def compare_page():
    """Trang giao diện so sánh giá nhà."""
    provinces = get_available_provinces()
    default_province = "Tp Hồ Chí Minh" if "Tp Hồ Chí Minh" in provinces else (provinces[0] if provinces else "")
    default_districts = get_districts_by_province(default_province) if default_province else []

    # Danh sách quận mặc định được đề xuất so sánh ban đầu
    initial_districts = []
    candidates = ["Quận 7", "Quận Bình Thạnh", "Thành phố Thủ Đức", "Quận Bình Tân"]
    for c in candidates:
        if c in default_districts:
            initial_districts.append(c)
    if not initial_districts and default_districts:
        initial_districts = default_districts[:3]

    return render_template(
        "compare.html",
        provinces=provinces,
        default_province=default_province,
        districts=default_districts,
        initial_districts=initial_districts
    )


@comparison_bp.route("/api/areas/provinces")
def api_provinces():
    """API lấy danh sách tỉnh/thành phố có trong dữ liệu."""
    provinces = get_available_provinces()
    return jsonify({
        "success": True,
        "provinces": provinces
    })


@comparison_bp.route("/api/areas/districts")
def api_districts():
    """API lấy danh sách quận/huyện theo tỉnh/thành phố."""
    province = request.args.get("province", "Tp Hồ Chí Minh")
    districts = get_districts_by_province(province)
    return jsonify({
        "success": True,
        "province": province,
        "districts": districts
    })


@comparison_bp.route("/api/areas/compare", methods=["POST"])
def api_compare():
    """API thực hiện so sánh dữ liệu bất động sản giữa các khu vực."""
    data = request.get_json(silent=True) or request.form.to_dict()
    province = data.get("province", "Tp Hồ Chí Minh").strip()
    districts = data.get("districts", [])

    if isinstance(districts, str):
        districts = [d.strip() for d in districts.split(",") if d.strip()]

    if not districts:
        return jsonify({
            "success": False,
            "error": "Vui lòng chọn ít nhất 1 khu vực (quận/huyện) để so sánh."
        }), 400

    try:
        result = compare_areas(province, districts)
        return jsonify(result)
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Lỗi xử lý so sánh: {e}"
        }), 500
