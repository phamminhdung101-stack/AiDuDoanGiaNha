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

# ── Bản đồ chuẩn hóa tên Thành phố & Địa phương (bí danh -> tên chuẩn) ───────────
CITY_CANONICAL_MAP = {
    # 1. TP. Hồ Chí Minh & Bí danh
    "hồ chí minh": "Tp Hồ Chí Minh",
    "tp hồ chí minh": "Tp Hồ Chí Minh",
    "tp. hồ chí minh": "Tp Hồ Chí Minh",
    "tp.hcm": "Tp Hồ Chí Minh",
    "tp hcm": "Tp Hồ Chí Minh",
    "thành phố hồ chí minh": "Tp Hồ Chí Minh",
    "tphcm": "Tp Hồ Chí Minh",
    "sài gòn": "Tp Hồ Chí Minh",
    "saigon": "Tp Hồ Chí Minh",
    "hcm": "Tp Hồ Chí Minh",

    # 2. Hà Nội
    "hà nội": "Hà Nội",
    "tp hà nội": "Hà Nội",
    "tp. hà nội": "Hà Nội",
    "thành phố hà nội": "Hà Nội",
    "hn": "Hà Nội",
    "thủ đô": "Hà Nội",
    "thủ đô hà nội": "Hà Nội",

    # 3. Đà Nẵng
    "đà nẵng": "Đà Nẵng",
    "tp đà nẵng": "Đà Nẵng",
    "tp. đà nẵng": "Đà Nẵng",
    "thành phố đà nẵng": "Đà Nẵng",
    "danang": "Đà Nẵng",

    # 4. Bà Rịa - Vũng Tàu
    "bà rịa - vũng tàu": "Bà Rịa - Vũng Tàu",
    "bà rịa vũng tàu": "Bà Rịa - Vũng Tàu",
    "vũng tàu": "Bà Rịa - Vũng Tàu",
    "tp vũng tàu": "Bà Rịa - Vũng Tàu",
    "tp. vũng tàu": "Bà Rịa - Vũng Tàu",
    "thành phố vũng tàu": "Bà Rịa - Vũng Tàu",
    "bà rịa": "Bà Rịa - Vũng Tàu",
    "tp bà rịa": "Bà Rịa - Vũng Tàu",
    "thành phố bà rịa": "Bà Rịa - Vũng Tàu",
    "brvt": "Bà Rịa - Vũng Tàu",
    "br-vt": "Bà Rịa - Vũng Tàu",

    # 5. Bình Dương
    "bình dương": "Bình Dương",
    "tỉnh bình dương": "Bình Dương",
    "thủ dầu một": "Bình Dương",
    "dĩ an": "Bình Dương",
    "thuận an": "Bình Dương",
    "bến cát": "Bình Dương",
    "tân uyên": "Bình Dương",

    # 6. Đồng Nai
    "đồng nai": "Đồng Nai",
    "tỉnh đồng nai": "Đồng Nai",
    "biên hòa": "Đồng Nai",
    "tp biên hòa": "Đồng Nai",
    "long thành": "Đồng Nai",
    "nhơn trạch": "Đồng Nai",

    # 7. Cần Thơ
    "cần thơ": "Cần Thơ",
    "tp cần thơ": "Cần Thơ",
    "tp. cần thơ": "Cần Thơ",
    "thành phố cần thơ": "Cần Thơ",

    # 8. Hải Phòng
    "hải phòng": "Hải Phòng",
    "tp hải phòng": "Hải Phòng",
    "thành phố hải phòng": "Hải Phòng",

    # 9. Khánh Hòa
    "khánh hòa": "Khánh Hòa",
    "nha trang": "Khánh Hòa",
    "tp nha trang": "Khánh Hòa",
    "thành phố nha trang": "Khánh Hòa",
    "cam ranh": "Khánh Hòa",

    # 10. Lâm Đồng
    "lâm đồng": "Lâm Đồng",
    "đà lạt": "Lâm Đồng",
    "tp đà lạt": "Lâm Đồng",
    "thành phố đà lạt": "Lâm Đồng",
    "bảo lộc": "Lâm Đồng",

    # 11. Quảng Ninh
    "quảng ninh": "Quảng Ninh",
    "hạ long": "Quảng Ninh",
    "tp hạ long": "Quảng Ninh",
    "cẩm phả": "Quảng Ninh",
    "uông bí": "Quảng Ninh",
    "móng cái": "Quảng Ninh",

    # 12. Thừa Thiên Huế
    "thừa thiên huế": "Thừa Thiên Huế",
    "huế": "Thừa Thiên Huế",
    "tp huế": "Thừa Thiên Huế",
    "thành phố huế": "Thừa Thiên Huế",

    # 13. Đắk Lắk
    "đắk lắk": "Đắk Lắk",
    "dak lak": "Đắk Lắk",
    "buôn ma thuột": "Đắk Lắk",
    "tp buôn ma thuột": "Đắk Lắk",

    # 14. Bình Định
    "bình định": "Bình Định",
    "quy nhơn": "Bình Định",
    "tp quy nhơn": "Bình Định",

    # 15. Kiên Giang
    "kiên giang": "Kiên Giang",
    "phú quốc": "Kiên Giang",
    "thành phố phú quốc": "Kiên Giang",
    "rạch giá": "Kiên Giang",
    "hà tiên": "Kiên Giang",

    # 16. Bình Thuận
    "bình thuận": "Bình Thuận",
    "phan thiết": "Bình Thuận",
    "mũi né": "Bình Thuận",

    # 17. Thanh Hóa
    "thanh hóa": "Thanh Hóa",
    "tp thanh hóa": "Thanh Hóa",
    "sầm sơn": "Thanh Hóa",

    # 18. Nghệ An
    "nghệ an": "Nghệ An",
    "vinh": "Nghệ An",
    "tp vinh": "Nghệ An",
    "cửa lò": "Nghệ An",

    # 19. Bắc Ninh
    "bắc ninh": "Bắc Ninh",
    "tp bắc ninh": "Bắc Ninh",
    "từ sơn": "Bắc Ninh",

    # 20. Hải Dương
    "hải dương": "Hải Dương",
    "tp hải dương": "Hải Dương",

    # 21. Hưng Yên
    "hưng yên": "Hưng Yên",
    "tp hưng yên": "Hưng Yên",

    # 22. Thái Nguyên
    "thái nguyên": "Thái Nguyên",
    "tp thái nguyên": "Thái Nguyên",

    # 23. Long An
    "long an": "Long An",
    "tân an": "Long An",

    # 24. Tiền Giang
    "tiền giang": "Tiền Giang",
    "mỹ tho": "Tiền Giang",

    # 25. Bến Tre
    "bến tre": "Bến Tre",
    "tp bến tre": "Bến Tre",

    # 26. Vĩnh Long
    "vĩnh long": "Vĩnh Long",

    # 27. Trà Vinh
    "trà vinh": "Trà Vinh",

    # 28. Hậu Giang
    "hậu giang": "Hậu Giang",
    "vị thanh": "Hậu Giang",

    # 29. Sóc Trăng
    "sóc trăng": "Sóc Trăng",

    # 30. Bạc Liêu
    "bạc liêu": "Bạc Liêu",

    # 31. Cà Mau
    "cà mau": "Cà Mau",

    # 32. An Giang
    "an giang": "An Giang",
    "long xuyên": "An Giang",
    "châu đốc": "An Giang",

    # 33. Đồng Tháp
    "đồng tháp": "Đồng Tháp",
    "cao lãnh": "Đồng Tháp",
    "sa đéc": "Đồng Tháp",

    # 34. Tây Ninh
    "tây ninh": "Tây Ninh",

    # 35. Bình Phước
    "bình phước": "Bình Phước",
    "đồng xoài": "Bình Phước",

    # 36. Ninh Thuận
    "ninh thuận": "Ninh Thuận",
    "phan rang": "Ninh Thuận",

    # 37. Phú Yên
    "phú yên": "Phú Yên",
    "tuy hòa": "Phú Yên",

    # 38. Quảng Nam
    "quảng nam": "Quảng Nam",
    "hội an": "Quảng Nam",
    "tam kỳ": "Quảng Nam",

    # 39. Quảng Ngãi
    "quảng ngãi": "Quảng Ngãi",

    # 40. Quảng Trị
    "quảng trị": "Quảng Trị",
    "đông hà": "Quảng Trị",

    # 41. Hà Nam
    "hà nam": "Hà Nam",
    "phủ lý": "Hà Nam",

    # 42. Nam Định
    "nam định": "Nam Định",

    # 43. Thái Bình
    "thái bình": "Thái Bình",

    # 44. Bắc Giang
    "bắc giang": "Bắc Giang",

    # 45. Phú Thọ
    "phú thọ": "Phú Thọ",
    "việt trì": "Phú Thọ",

    # 46. Lào Cai
    "lào cai": "Lào Cai",
    "sa pa": "Lào Cai",
    "sapa": "Lào Cai",

    # 47. Yên Bái
    "yên bái": "Yên Bái",

    # 48. Hòa Bình
    "hòa bình": "Hòa Bình",

    # 49. Ninh Bình
    "ninh bình": "Ninh Bình",

    # 50. Kon Tum
    "kon tum": "Kon Tum",

    # 51. Gia Lai
    "gia lai": "Gia Lai",
    "pleiku": "Gia Lai",

    # 52. Đắk Nông
    "đắk nông": "Đắk Nông",
    "gia nghĩa": "Đắk Nông",

    # Các tỉnh thuộc Việt Nam nằm ngoài danh mục huấn luyện 52 tỉnh:
    "điện biên": "Điện Biên",
    "điện biên phủ": "Điện Biên",
    "sơn la": "Sơn La",
    "lai châu": "Lai Châu",
    "hà giang": "Hà Giang",
    "cao bằng": "Cao Bằng",
    "bắc kạn": "Bắc Kạn",
    "tuyên quang": "Tuyên Quang",
    "lạng sơn": "Lạng Sơn",
    "vĩnh phúc": "Vĩnh Phúc",
    "vĩnh yên": "Vĩnh Phúc",
    "quảng bình": "Quảng Bình",
    "đồng hới": "Quảng Bình",
    "hà tĩnh": "Hà Tĩnh",
}

# Danh mục 63 tỉnh/thành phố chính thức của Việt Nam để xác định địa danh hợp lệ
ALL_VIETNAMESE_PROVINCES = {
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh",
    "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
    "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
    "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An", "Nam Định",
    "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình",
    "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La",
    "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang",
    "Tp Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
}

# Bản đồ ánh xạ Quận/Huyện đặc trưng về Thành phố trực thuộc
DISTRICT_TO_CITY_MAP = {
    # TP. Hồ Chí Minh
    "quận 1": "Tp Hồ Chí Minh", "quận 2": "Tp Hồ Chí Minh", "quận 3": "Tp Hồ Chí Minh",
    "quận 4": "Tp Hồ Chí Minh", "quận 5": "Tp Hồ Chí Minh", "quận 6": "Tp Hồ Chí Minh",
    "quận 7": "Tp Hồ Chí Minh", "quận 8": "Tp Hồ Chí Minh", "quận 9": "Tp Hồ Chí Minh",
    "quận 10": "Tp Hồ Chí Minh", "quận 11": "Tp Hồ Chí Minh", "quận 12": "Tp Hồ Chí Minh",
    "thành phố thủ đức": "Tp Hồ Chí Minh", "thủ đức": "Tp Hồ Chí Minh",
    "bình thạnh": "Tp Hồ Chí Minh", "quận bình thạnh": "Tp Hồ Chí Minh",
    "gò vấp": "Tp Hồ Chí Minh", "quận gò vấp": "Tp Hồ Chí Minh",
    "tân bình": "Tp Hồ Chí Minh", "quận tân bình": "Tp Hồ Chí Minh",
    "tân phú": "Tp Hồ Chí Minh", "quận tân phú": "Tp Hồ Chí Minh",
    "phú nhuận": "Tp Hồ Chí Minh", "quận phú nhuận": "Tp Hồ Chí Minh",
    "bình tân": "Tp Hồ Chí Minh", "quận bình tân": "Tp Hồ Chí Minh",
    "huyện bình chánh": "Tp Hồ Chí Minh", "bình chánh": "Tp Hồ Chí Minh",
    "huyện nhà bè": "Tp Hồ Chí Minh", "nhà bè": "Tp Hồ Chí Minh",
    "huyện hóc môn": "Tp Hồ Chí Minh", "hóc môn": "Tp Hồ Chí Minh",
    "huyện củ chi": "Tp Hồ Chí Minh", "củ chi": "Tp Hồ Chí Minh",
    "huyện cần giờ": "Tp Hồ Chí Minh", "cần giờ": "Tp Hồ Chí Minh",

    # Hà Nội
    "ba đình": "Hà Nội", "quận ba đình": "Hà Nội",
    "hoàn kiếm": "Hà Nội", "quận hoàn kiếm": "Hà Nội",
    "tây hồ": "Hà Nội", "quận tây hồ": "Hà Nội",
    "long biên": "Hà Nội", "quận long biên": "Hà Nội",
    "cầu giấy": "Hà Nội", "quận cầu giấy": "Hà Nội",
    "đống đa": "Hà Nội", "quận đống đa": "Hà Nội",
    "hai bà trưng": "Hà Nội", "quận hai bà trưng": "Hà Nội",
    "hoàng mai": "Hà Nội", "quận hoàng mai": "Hà Nội",
    "thanh xuân": "Hà Nội", "quận thanh xuân": "Hà Nội",
    "hà đông": "Hà Nội", "quận hà đông": "Hà Nội",
    "bắc từ liêm": "Hà Nội", "quận bắc từ liêm": "Hà Nội",
    "nam từ liêm": "Hà Nội", "quận nam từ liêm": "Hà Nội",
    "thanh trì": "Hà Nội", "gia lâm": "Hà Nội", "đông anh": "Hà Nội", "sóc sơn": "Hà Nội",

    # Đà Nẵng
    "hải châu": "Đà Nẵng", "quận hải châu": "Đà Nẵng",
    "sơn trà": "Đà Nẵng", "quận sơn trà": "Đà Nẵng",
    "ngũ hành sơn": "Đà Nẵng", "quận ngũ hành sơn": "Đà Nẵng",
    "thanh khê": "Đà Nẵng", "quận thanh khê": "Đà Nẵng",
    "liên chiểu": "Đà Nẵng", "quận liên chiểu": "Đà Nẵng",
    "cẩm lệ": "Đà Nẵng", "quận cẩm lệ": "Đà Nẵng",
    "hòa vang": "Đà Nẵng", "huyện hòa vang": "Đà Nẵng",
}

INVALID_TOKENS = {
    "", "unknown", "none", "nan", "null", "undefined", "chưa rõ", "chua ro", "n/a",
    "không có xã/phường", "không có phường/xã", "không có xã", "không có phường", "chưa có"
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

def validate_location(city: Any, ward: Any, catalog: dict[str, list[str]], allow_unknown_ward: bool = False) -> tuple[str, str]:
    """
    Kiểm tra tính hợp lệ của thành phố và phường/xã dựa trên danh mục địa điểm của mô hình.
    Trả về (normalized_city, normalized_ward) nếu hợp lệ.
    Ném ValueError với thông báo tiếng Việt rõ ràng nếu không hợp lệ.
    Nếu allow_unknown_ward=True: cho phép phường/xã rỗng để định giá theo baseline của thành phố.
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
        valid_cities = ", ".join(f"'{c}'" for c in sorted(catalog.keys()))
        raise ValueError(
            f"Thành phố '{city}' không tồn tại trong danh mục của mô hình. "
            f"Các thành phố hợp lệ: {valid_cities}."
        )

    norm_ward = normalize_ward(ward)
    if not norm_ward:
        if allow_unknown_ward:
            return matched_city, ""
        raise ValueError(f"Vui lòng chọn Phường / Xã thuộc {matched_city}.")

    wards_in_city = catalog[matched_city]

    # Tìm phường trong thành phố
    matched_ward = None
    for w in wards_in_city:
        if w.lower() == norm_ward.lower():
            matched_ward = w
            break

    if not matched_ward:
        if allow_unknown_ward:
            # Nếu người dùng nhập tên phường không có trong danh mục, cho phép fallback về mức chung thành phố
            return matched_city, ""

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

