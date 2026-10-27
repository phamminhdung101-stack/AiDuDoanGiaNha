"""
location_service.py — Quản lý danh mục địa điểm, chuẩn hóa và xác thực dữ liệu.

Chức năng:
- Chuẩn hóa tên thành phố, phường/xã, tên đường đồng nhất giữa huấn luyện và dự đoán.
- Tạo danh mục địa điểm động từ DataFrame đã làm sạch sau khi train.
- Giữ đúng quan hệ Thành phố -> Phường/Xã (không gộp chung vào tất cả thành phố).
- Kiểm tra tính hợp lệ khi dự đoán ở backend với thông báo tiếng Việt rõ ràng.
- Đánh giá chất lượng dữ liệu đường và cung cấp cơ chế xác thực đường cắm rút (pluggable validator).
- Lưu trữ và cập nhật danh mục địa điểm an toàn, nguyên tử (atomic update).
"""
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT_DIR / "models" / "location_catalog.json"
STREET_CATALOG_PATH = ROOT_DIR / "models" / "street_catalog.json"
TRAINED_DATA_PATH = ROOT_DIR / "models" / "trained_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "trained_model.pkl"

# ── Quy tắc chuẩn hóa dấu tiếng Việt (bộ gõ mới vs cũ) ────────────────────────
VIETNAMESE_ACCENT_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}

# ── Bản đồ chuẩn hóa tên Thành phố phổ biến (bí danh -> tên chuẩn) ───────────
CITY_CANONICAL_MAP = {
    "hồ chí minh": "Tp Hồ Chí Minh",
    "tp hồ chí minh": "Tp Hồ Chí Minh",
    "tp. hồ chí minh": "Tp Hồ Chí Minh",
    "tp.hcm": "Tp Hồ Chí Minh",
    "tp hcm": "Tp Hồ Chí Minh",
    "thành phố hồ chí minh": "Tp Hồ Chí Minh",
    "tphcm": "Tp Hồ Chí Minh",
    "hà nội": "Hà Nội",
    "tp hà nội": "Hà Nội",
    "tp. hà nội": "Hà Nội",
    "thành phố hà nội": "Hà Nội",
    "đà nẵng": "Đà Nẵng",
    "tp đà nẵng": "Đà Nẵng",
    "tp. đà nẵng": "Đà Nẵng",
    "thành phố đà nẵng": "Đà Nẵng",
    "cần thơ": "Cần Thơ",
    "tp cần thơ": "Cần Thơ",
    "tp. cần thơ": "Cần Thơ",
    "thành phố cần thơ": "Cần Thơ",
    "bình dương": "Bình Dương",
    "tỉnh bình dương": "Bình Dương",
    "hải phòng": "Hải Phòng",
    "tp hải phòng": "Hải Phòng",
    "thành phố hải phòng": "Hải Phòng",
}

INVALID_TOKENS = {
    "", "unknown", "none", "nan", "null", "undefined", "chưa rõ", "chua ro", "n/a"
}

_cached_catalog: dict[str, list[str]] | None = None


def standardize_vietnamese_accents(text: str) -> str:
    """Chuẩn hóa vị trí dấu thanh tiếng Việt về dạng thống nhất."""
    if not text:
        return ""
    for k, v in VIETNAMESE_ACCENT_MAP.items():
        text = text.replace(k, v)
        text = text.replace(k.title(), v.title())
        text = text.replace(k.upper(), v.upper())
    return text


def normalize_string(val: Any) -> str:
    """
    Chuẩn hóa khoảng trắng, Unicode NFC, bỏ ký tự thừa và lọc các giá trị rỗng/unknown.
    """
    if val is None or pd.isna(val):
        return ""
    text = str(val)
    text = unicodedata.normalize("NFC", text)
    text = standardize_vietnamese_accents(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in INVALID_TOKENS:
        return ""
    return text


def normalize_city(city: Any) -> str:
    """
    Chuẩn hóa tên thành phố: loại bỏ khoảng trắng thừa, chuẩn hóa Unicode và ánh xạ bí danh.
    Không tự ý gộp các thành phố khác nhau.
    """
    cleaned = normalize_string(city)
    if not cleaned:
        return ""
    lower = cleaned.lower()
    return CITY_CANONICAL_MAP.get(lower, cleaned)


def normalize_ward(ward: Any) -> str:
    """
    Chuẩn hóa tên phường/xã/thị trấn: loại bỏ khoảng trắng thừa, chuẩn hóa Unicode.
    Giữ nguyên các hậu tố phân biệt (như 'Quận 9 cũ', 'A', 'B') để không tự ý gộp địa danh khác nhau.
    """
    return normalize_string(ward)


def normalize_street(street: Any) -> str:
    """Chuẩn hóa tên đường/phố."""
    return normalize_string(street)


# ── Tạo và quản lý danh mục địa điểm ────────────────────────────────────────

def build_location_catalog(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Tạo danh mục quan hệ thành phố -> các phường/xã từ DataFrame đã làm sạch.
    - Giữ đúng quan hệ thành phố -> phường/xã xuất hiện trong thành phố đó.
    - Không gộp một phường vào tất cả thành phố.
    - Loại bỏ giá trị rỗng/unknown và trùng lặp.
    - Nếu không có địa điểm hợp lệ, báo lỗi rõ ràng.
    """
    if "location" not in df.columns or "ward" not in df.columns:
        raise ValueError("Tập dữ liệu thiếu cột 'location' hoặc 'ward'.")

    catalog: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        c = normalize_city(row.get("location"))
        w = normalize_ward(row.get("ward"))
        if c and w:
            if c not in catalog:
                catalog[c] = set()
            catalog[c].add(w)

    # Chuyển set sang danh sách đã sắp xếp
    sorted_catalog: dict[str, list[str]] = {}
    for city in sorted(catalog.keys()):
        wards = sorted(list(catalog[city]))
        if wards:
            sorted_catalog[city] = wards

    if not sorted_catalog:
        raise ValueError(
            "Tập dữ liệu sau khi làm sạch không chứa địa điểm (thành phố / phường xã) hợp lệ để tạo danh mục."
        )

    return sorted_catalog


def save_location_catalog(catalog: dict[str, list[str]], catalog_path: Path = CATALOG_PATH) -> None:
    """Lưu danh mục địa điểm ra file JSON nguyên tử (atomic write)."""
    catalog_path = Path(catalog_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{catalog_path.name}.",
            suffix=".tmp",
            dir=catalog_path.parent,
        )
        os.close(fd)
        temp_path = Path(temp_name)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, catalog_path)
        temp_path = None
        global _cached_catalog
        _cached_catalog = catalog
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def clear_location_cache() -> None:
    """Xóa bộ đệm danh mục địa điểm."""
    global _cached_catalog
    _cached_catalog = None


def get_active_location_catalog(force_reload: bool = False) -> dict[str, list[str]]:
    """
    Lấy danh mục địa điểm đang hoạt động:
    1. Trả về từ bộ nhớ đệm nếu có.
    2. Nếu không, đọc từ models/location_catalog.json.
    3. Nếu chưa có file JSON, đọc từ models/trained_model.pkl nếu bundle có lưu.
    4. Nếu vẫn chưa có, trích xuất từ models/trained_data.csv nếu tồn tại.
    """
    global _cached_catalog
    if _cached_catalog is not None and not force_reload:
        return _cached_catalog

    # 1. Đọc từ JSON
    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    _cached_catalog = data
                    return _cached_catalog
        except Exception:
            pass

    # 2. Đọc từ model bundle .pkl
    if MODEL_PATH.exists():
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and bundle.get("location_catalog"):
                catalog = bundle["location_catalog"]
                save_location_catalog(catalog)
                _cached_catalog = catalog
                return _cached_catalog
        except Exception:
            pass

    # 3. Trích xuất từ trained_data.csv (dành cho mô hình cũ đã train trước khi cập nhật)
    if TRAINED_DATA_PATH.exists():
        try:
            df = pd.read_csv(TRAINED_DATA_PATH)
            catalog = build_location_catalog(df)
            save_location_catalog(catalog)
            _cached_catalog = catalog
            return _cached_catalog
        except Exception:
            pass

    return {}


# ── Kiểm tra tính hợp lệ khi dự đoán ─────────────────────────────────────────

def validate_location(city: Any, ward: Any, catalog: dict[str, list[str]]) -> tuple[str, str]:
    """
    Kiểm tra tính hợp lệ của thành phố và phường/xã dựa trên danh mục địa điểm của mô hình.
    Trả về (normalized_city, normalized_ward) nếu hợp lệ.
    Ném ValueError với thông báo tiếng Việt rõ ràng nếu không hợp lệ.
    """
    if not catalog:
        raise ValueError(
            "Mô hình chưa có danh mục địa điểm hợp lệ. "
            "Quản trị viên vui lòng huấn luyện lại mô hình với tập dữ liệu chuẩn."
        )

    norm_city = normalize_city(city)
    if not norm_city:
        raise ValueError("Vui lòng chọn Tỉnh / Thành phố.")

    # Tìm thành phố trong danh mục (so sánh không phân biệt hoa thường)
    matched_city = None
    for c in catalog.keys():
        if c.lower() == norm_city.lower():
            matched_city = c
            break

    if not matched_city:
        valid_cities = ", ".join(f"'{c}'" for c in catalog.keys())
        raise ValueError(
            f"Thành phố '{city}' không tồn tại trong danh mục của mô hình. "
            f"Các thành phố hợp lệ: {valid_cities}."
        )

    norm_ward = normalize_ward(ward)
    if not norm_ward:
        raise ValueError(f"Vui lòng chọn Phường / Xã thuộc {matched_city}.")

    wards_in_city = catalog[matched_city]

    # Tìm phường trong thành phố
    matched_ward = None
    for w in wards_in_city:
        if w.lower() == norm_ward.lower():
            matched_ward = w
            break

    if not matched_ward:
        # Kiểm tra xem phường này có thuộc thành phố khác trong danh mục hay không
        other_cities = [
            c for c, w_list in catalog.items()
            if any(w.lower() == norm_ward.lower() for w in w_list)
        ]
        if other_cities:
            raise ValueError(
                f"Phường/xã '{ward}' thuộc {', '.join(other_cities)}, không thuộc thành phố '{matched_city}'. "
                f"Vui lòng chọn đúng phường/xã thuộc {matched_city}."
            )
        raise ValueError(
            f"Phường/xã '{ward}' không tồn tại trong danh mục của thành phố '{matched_city}'. "
            f"Vui lòng chọn phường/xã hợp lệ từ danh mục."
        )

    return matched_city, matched_ward


# ── Xác thực đường & Đánh giá dữ liệu đường ─────────────────────────────────

_cached_street_catalog: dict[str, dict[str, list[str]]] | None = None


def clear_street_catalog_cache() -> None:
    """Xóa bộ đệm danh mục đường."""
    global _cached_street_catalog
    _cached_street_catalog = None


def load_street_catalog(catalog_path: Path | str | None = None) -> dict[str, dict[str, list[str]]]:
    """
    Đọc danh mục đường có cấu trúc từ file JSON (mặc định models/street_catalog.json).
    - Nếu file không tồn tại: trả về dict rỗng {}.
    - Nếu file tồn tại nhưng rỗng: trả về dict rỗng {}.
    - Nếu file lỗi cú pháp JSON hoặc sai cấu trúc: ném ValueError rõ ràng (không âm thầm coi như danh mục hợp lệ).
    - Chuẩn hóa khoảng trắng và Unicode cho city, ward, street theo các helper hiện có.
    """
    path = Path(catalog_path) if catalog_path else STREET_CATALOG_PATH
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read().strip()
            if not raw_content:
                return {}
            data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Lỗi đọc file danh mục đường '{path.name}': nội dung JSON không hợp lệ ({exc})."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Lỗi đọc file danh mục đường '{path.name}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Lỗi cấu trúc file danh mục đường '{path.name}': dữ liệu gốc phải là một JSON object/dictionary."
        )

    normalized: dict[str, dict[str, list[str]]] = {}
    for city_key, ward_map in data.items():
        norm_city_key = normalize_city(city_key)
        if not norm_city_key:
            continue
        if not isinstance(ward_map, dict):
            raise ValueError(
                f"Lỗi cấu trúc file danh mục đường: danh sách phường của thành phố '{city_key}' phải là một dictionary."
            )
        normalized[norm_city_key] = {}
        for ward_key, street_list in ward_map.items():
            norm_ward_key = normalize_ward(ward_key)
            if not norm_ward_key:
                continue
            if not isinstance(street_list, list):
                raise ValueError(
                    f"Lỗi cấu trúc file danh mục đường: danh sách đường của phường '{ward_key}' phải là một mảng/list."
                )
            cleaned_streets = []
            for s in street_list:
                norm_s = normalize_street(s)
                if norm_s:
                    cleaned_streets.append(norm_s)
            normalized[norm_city_key][norm_ward_key] = cleaned_streets

    return normalized


def get_active_street_catalog(force_reload: bool = False, catalog_path: Path | str | None = None) -> dict[str, dict[str, list[str]]]:
    """
    Lấy danh mục đường đang hoạt động (có bộ đệm bộ nhớ).
    """
    global _cached_street_catalog
    if _cached_street_catalog is not None and not force_reload and catalog_path is None:
        return _cached_street_catalog

    catalog = load_street_catalog(catalog_path)
    if catalog_path is None:
        _cached_street_catalog = catalog
    return catalog


class StreetValidator:
    """
    Cơ chế xác thực quan hệ đường -> phường/xã.
    - Chỉ xác nhận đường thuộc phường khi có danh mục đường-phường đáng tin cậy.
    - Nếu danh mục chưa tồn tại hoặc rỗng:
        + Không tuyên bố đã xác thực.
        + Không chặn hợp lệ một cách tùy tiện.
        + Trả về trạng thái "chưa đối chiếu danh mục đường" kèm giải thích.
    """
    def __init__(self, street_catalog: dict[str, dict[str, list[str]]] | None = None):
        # Cấu trúc street_catalog: { city: { ward: [street1, street2, ...] } }
        if street_catalog is None:
            self.street_catalog = get_active_street_catalog()
        else:
            self.street_catalog = street_catalog

    @property
    def is_active(self) -> bool:
        """Kiểm tra xem hệ thống có danh mục đường đáng tin cậy để kích hoạt xác thực hay không."""
        if not self.street_catalog:
            return False
        return any(
            bool(ward_dict) and any(bool(streets) for streets in ward_dict.values())
            for ward_dict in self.street_catalog.values()
        )

    def validate(self, city: str, ward: str, street: Any) -> dict[str, Any]:
        """
        Xác thực tên đường:
        - Nếu street để trống: hợp lệ (đường không bắt buộc).
        - Nếu có danh mục đáng tin cậy:
            + Nếu đường có trong danh mục của phường đã chọn: hợp lệ, đánh dấu đã xác thực.
            + Nếu đường có trong danh mục nhưng thuộc phường khác: từ chối, báo rõ thuộc phường nào.
            + Nếu đường không có trong danh mục của phường: từ chối, báo tên đường không có trong danh mục của phường đã chọn.
        - Nếu không có danh mục đáng tin cậy (hoặc rỗng):
            Không tuyên bố đã xác thực, không chặn tùy tiện, trả về trạng thái "unverified".
        """
        norm_street = normalize_street(street)
        if not norm_street:
            return {
                "street": "",
                "status": "empty",
                "verified": False,
                "message": "",
            }

        if not self.is_active:
            # File danh mục chưa tồn tại hoặc rỗng
            return {
                "street": norm_street,
                "status": "unverified",
                "verified": False,
                "message": (
                    "Hệ thống chưa bật xác thực đường–phường do danh mục đường chưa được cấu hình hoặc rỗng."
                ),
            }

        # Tìm thành phố trong danh mục (so sánh không phân biệt hoa thường và chuẩn hóa)
        target_city_norm = normalize_city(city).lower()
        matched_city_wards: dict[str, list[str]] = {}
        for c_key, w_dict in self.street_catalog.items():
            if normalize_city(c_key).lower() == target_city_norm:
                matched_city_wards = w_dict
                break

        # Tìm phường trong thành phố đã chọn
        target_ward_norm = normalize_ward(ward).lower()
        streets_in_target_ward: list[str] = []
        for w_key, s_list in matched_city_wards.items():
            if normalize_ward(w_key).lower() == target_ward_norm:
                streets_in_target_ward = s_list
                break

        # 1. Kiểm tra xem đường có trong phường đã chọn không
        matched_street = next(
            (s for s in streets_in_target_ward if normalize_street(s).lower() == norm_street.lower()),
            None
        )
        if matched_street:
            return {
                "street": matched_street,
                "status": "verified",
                "verified": True,
                "message": f"Đường '{matched_street}' đã được xác thực thuộc {ward}, {city}.",
            }

        # 2. Nếu không có trong phường đã chọn: kiểm tra xem có thuộc phường khác trong cùng thành phố không
        found_in_same_city_wards = []
        for other_w_key, s_list in matched_city_wards.items():
            if normalize_ward(other_w_key).lower() == target_ward_norm:
                continue
            for s in s_list:
                if normalize_street(s).lower() == norm_street.lower():
                    found_in_same_city_wards.append(other_w_key)
                    break

        if found_in_same_city_wards:
            raise ValueError(
                f"Đường '{street}' thuộc {', '.join(found_in_same_city_wards)}, "
                f"không thuộc phường/xã '{ward}' ({city})."
            )

        # 3. Kiểm tra xem có thuộc thành phố khác trong danh mục không
        found_in_other_cities = []
        for other_c_key, other_w_dict in self.street_catalog.items():
            if normalize_city(other_c_key).lower() == target_city_norm:
                continue
            for other_w_key, s_list in other_w_dict.items():
                for s in s_list:
                    if normalize_street(s).lower() == norm_street.lower():
                        found_in_other_cities.append(f"{other_w_key} ({other_c_key})")
                        break

        if found_in_other_cities:
            raise ValueError(
                f"Đường '{street}' thuộc {', '.join(found_in_other_cities)}, "
                f"không thuộc phường/xã '{ward}' ({city})."
            )

        # 4. Đường không có trong danh mục của phường đã chọn
        raise ValueError(
            f"Đường '{street}' không có trong danh mục đường của phường/xã '{ward}' ({city})."
        )


def get_street_data_assessment(df: pd.DataFrame) -> dict[str, Any]:
    """
    Xác minh xem DataFrame/dataset thực tế có đủ cột hoặc dữ liệu địa chỉ để lập quan hệ đường -> phường hay không.
    Trả về báo cáo chi tiết về giới hạn dữ liệu và đề xuất dữ liệu cần bổ sung.
    """
    total_rows = len(df)
    has_address = "address" in df.columns
    has_street = "street" in df.columns

    address_series = df["address"] if has_address else pd.Series([""] * total_rows)
    street_series = df["street"] if has_street else pd.Series([""] * total_rows)

    clean_addr = address_series.fillna("").astype(str).str.strip()
    clean_street = street_series.fillna("").astype(str).str.strip()

    valid_addr_count = int((clean_addr != "").sum())
    valid_street_count = int((clean_street != "").sum())

    effective_count = max(valid_addr_count, valid_street_count)
    coverage_pct = round((effective_count / total_rows * 100), 2) if total_rows > 0 else 0.0

    # Đánh giá: chỉ coi là đủ nếu có trường đường có cấu trúc và độ bao phủ >= 80%
    is_sufficient = (effective_count >= 50) and (coverage_pct >= 80.0)

    proposal = (
        "Để kích hoạt xác thực quan hệ đường -> phường, cần bổ sung một trong hai nguồn dữ liệu:\n"
        "1. Danh mục hành chính chính thức ánh xạ đầy đủ danh sách các tuyến đường theo từng Phường/Xã "
        "của Thành phố (từ cơ sở dữ liệu địa giới hành chính hoặc bản đồ số chuẩn hóa).\n"
        "2. Tập dữ liệu bất động sản có cột 'street' riêng biệt đã được làm sạch và chuẩn hóa, "
        "với độ phủ thông tin đạt tối thiểu 80-90% các dòng dữ liệu."
    )

    return {
        "total_rows": total_rows,
        "valid_address_count": valid_addr_count,
        "valid_street_count": valid_street_count,
        "effective_count": effective_count,
        "coverage_percentage": coverage_pct,
        "is_sufficient_for_validation": is_sufficient,
        "limitation_reason": (
            f"Tập dữ liệu chỉ có {effective_count}/{total_rows} dòng ({coverage_pct}%) có thông tin địa chỉ/đường. "
            f"Hầu hết các giá trị là chuỗi địa chỉ tự do hoặc rỗng (99%+ khuyết thiếu), không có cấu trúc "
            f"ánh xạ tuyến đường sang từng phường/xã cụ thể."
        ),
        "proposal": proposal,
    }

