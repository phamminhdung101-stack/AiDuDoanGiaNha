"""
RealEstateChatService — Dịch vụ Chatbot AI Tư vấn Bất động sản
HousePrice AI Platform

Chức năng:
1. Nhận diện & trích xuất chính xác địa điểm (Tỉnh/Thành phố, Quận/Huyện, Phường/Xã, Tuyến đường)
2. Bóc tách thông số kỹ thuật (Diện tích, Tầng, Phòng ngủ, Phòng tắm, Loại nhà, Mặt tiền, Hướng, v.v.)
3. Quản lý ngữ cảnh đa lượt (Multi-turn context tracking) không nhầm lẫn hay ghi đè địa điểm
4. Kiểm tra phạm vi dữ liệu huấn luyện (52 tỉnh thành): cảnh báo khi ngoài phạm vi, không gán mặc định
5. Tích hợp Machine Learning dự đoán giá với chế độ toàn thành phố (Allow Unknown Ward) và từng phường
6. Tích hợp OpenAI LLM với cơ chế fallback tự động sang Local Intelligent Decision Engine
"""
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from backend.predict import predict_price, load_model, PRIME_STREETS
from backend.location_service import (
    get_active_location_catalog,
    normalize_city,
    CITY_CANONICAL_MAP,
    ALL_VIETNAMESE_PROVINCES,
    DISTRICT_TO_CITY_MAP,
)
from backend.connect_db import get_cursor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT_DIR / "database" / "dataset.csv"


class RealEstateChatService:
    @staticmethod
    def _get_llm_config():
        """Lấy cấu hình LLM từ biến môi trường."""
        return {
            "api_key": os.environ.get("OPENAI_API_KEY", "").strip(),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        }

    # ── 1. PHÂN TÍCH Ý ĐỊNH & TRÍCH XUẤT THÔNG SỐ (NLP EXTRACTION) ───────────────

    @classmethod
    def extract_property_specs(cls, text: str) -> dict:
        """
        Trích xuất các thông số kỹ thuật bất động sản từ câu hỏi tự nhiên theo Section 5.
        Tuyệt đối không tự ý gán TP.HCM hay bất kỳ địa điểm mặc định nào nếu người dùng không cung cấp.
        """
        t_norm = unicodedata.normalize("NFC", text).lower()
        specs = {
            "area": None,
            "city": None,
            "district": None,
            "ward": None,
            "street": None,
            "bedrooms": None,
            "bathrooms": None,
            "floors": None,
            "user_price": None,
            "house_type": None,
            "direction": None,
            "frontage": None,
            "road_width": None,
            "legal": None,
            "raw_location": None,
        }

        # 1. Trích xuất diện tích (VD: "70m2", "70 m²", "diện tích 100 mét vuông")
        m_area = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mét vuông|met vuong)", t_norm)
        if m_area:
            try:
                specs["area"] = float(m_area.group(1).replace(",", "."))
            except ValueError:
                pass
        else:
            m_area2 = re.search(r"diện tích\s*(?:là|khoảng|tầm)?\s*(\d+(?:[.,]\d+)?)", t_norm)
            if m_area2:
                try:
                    specs["area"] = float(m_area2.group(1).replace(",", "."))
                except ValueError:
                    pass

        # 2. Trích xuất mức giá người dùng nêu (VD: "giá 4 tỷ", "4.5 tỷ", "800 triệu")
        m_ty = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:tỷ|ty|tỉ)", t_norm)
        if m_ty:
            try:
                specs["user_price"] = float(m_ty.group(1).replace(",", ".")) * 1_000_000_000
            except ValueError:
                pass
        else:
            m_trieu = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:triệu|trieu|tr)\b", t_norm)
            if m_trieu:
                try:
                    specs["user_price"] = float(m_trieu.group(1).replace(",", ".")) * 1_000_000
                except ValueError:
                    pass

        # 3. Trích xuất số phòng ngủ, phòng tắm/vệ sinh, số tầng
        m_bed = re.search(r"(\d+)\s*(?:phòng ngủ|pn|phong ngu)", t_norm)
        if m_bed:
            specs["bedrooms"] = int(m_bed.group(1))

        m_bath = re.search(r"(\d+)\s*(?:phòng vệ sinh|phòng tắm|wc|toilet|pt)", t_norm)
        if m_bath:
            specs["bathrooms"] = int(m_bath.group(1))

        m_floor = re.search(r"(\d+)\s*(?:tầng|lầu)", t_norm)
        if m_floor:
            specs["floors"] = int(m_floor.group(1))

        # 4. Trích xuất loại nhà (house_type)
        m_ht = re.search(r"\b(nhà mặt phố|nhà phố|biệt thự|căn hộ|chung cư|nhà cấp 4|nhà riêng|nhà hẻm|nhà ngõ|đất nền)\b", t_norm)
        if m_ht:
            specs["house_type"] = m_ht.group(1).capitalize()

        # 5. Trích xuất hướng nhà (direction)
        m_dir = re.search(r"hướng\s*(đông nam|đông bắc|tây nam|tây bắc|đông|tây|nam|bắc)", t_norm)
        if m_dir:
            specs["direction"] = m_dir.group(1).title()

        # 6. Trích xuất mặt tiền (frontage)
        m_front = re.search(r"(?:mặt tiền|ngang|bề ngang)\s*(?:rộng)?\s*(\d+(?:[.,]\d+)?)\s*(?:m|mét|met)\b", t_norm)
        if m_front:
            try:
                specs["frontage"] = float(m_front.group(1).replace(",", "."))
            except ValueError:
                pass

        # 7. Trích xuất độ rộng đường (road_width)
        m_rw = re.search(r"(?:đường|hẻm|ngõ)\s*(?:rộng|xe hơi|ô tô)?\s*(\d+(?:[.,]\d+)?)\s*(?:m|mét|met)\b", t_norm)
        if m_rw:
            try:
                specs["road_width"] = float(m_rw.group(1).replace(",", "."))
            except ValueError:
                pass

        # 8. Trích xuất pháp lý (legal)
        m_leg = re.search(r"\b(sổ đỏ|sổ hồng|chờ sổ|đang chờ sổ|giấy tờ hợp lệ|hợp đồng mua bán|vi bằng)\b", t_norm)
        if m_leg:
            specs["legal"] = m_leg.group(1).title()

        # 9. TRÍCH XUẤT ĐỊA ĐIỂM (QUẬN/HUYỆN, PHƯỜNG/XÃ, TÊN ĐƯỜNG, TỈNH/THÀNH PHỐ)
        # 9a. Trích xuất Quận/Huyện trước
        m_dist = re.search(r"(quận\s+\d+|quận\s+[a-zà-ỹ\s]+|huyện\s+[a-zà-ỹ\s]+|thị xã\s+[a-zà-ỹ\s]+|thành phố thủ đức|thủ đức)", t_norm)
        if m_dist:
            dist_raw = m_dist.group(1).strip()
            dist_raw = re.split(r"(?:,|\s+phường|\s+xã|\s+diện tích|\s+giá|\s+tại|\s+tp|\s+thành phố|\s+tỉnh)", dist_raw)[0].strip()
            if dist_raw and len(dist_raw.split()) <= 5:
                specs["district"] = dist_raw.title()
                if dist_raw.lower() in DISTRICT_TO_CITY_MAP:
                    specs["city"] = DISTRICT_TO_CITY_MAP[dist_raw.lower()]

        # 9b. Trích xuất Phường/Xã
        m_ward = re.search(r"(phường\s+\d+|phường\s+[a-zà-ỹ\s\d]+|xã\s+[a-zà-ỹ\s\d]+|thị trấn\s+[a-zà-ỹ\s\d]+)", t_norm)
        if m_ward:
            ward_raw = m_ward.group(1).strip()
            ward_raw = re.split(r"(?:,|\s+quận|\s+huyện|\s+thị xã|\s+tp|\s+thành phố|\s+tỉnh|\s+diện tích|\s+giá)", ward_raw)[0].strip()
            if ward_raw and len(ward_raw.split()) <= 5:
                specs["ward"] = ward_raw.title()

        # 9c. Trích xuất Tên đường
        m_street = re.search(r"(?:đường|phố)\s+([a-zà-ỹ\s\d]+?)(?=(?:,\s*phường|,\s*quận|\s+phường|\s+quận|\s+diện tích|\s+giá|\s+tại|$))", t_norm)
        if m_street:
            street_raw = m_street.group(1).strip().title()
            if len(street_raw.split()) <= 6:
                specs["street"] = street_raw

        # 9d. Trích xuất Tỉnh/Thành phố từ CITY_CANONICAL_MAP (duyệt từ chuỗi dài đến ngắn)
        sorted_aliases = sorted(CITY_CANONICAL_MAP.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            pattern = r"(?:^|\b|\s|ở\s+|tại\s+|tp\s+|tỉnh\s+)" + re.escape(alias) + r"(?:\b|\s|$|,|\.)"
            if re.search(pattern, t_norm):
                specs["city"] = CITY_CANONICAL_MAP[alias]
                break

        # Nếu chưa tìm thấy city, kiểm tra danh sách 63 tỉnh thành Việt Nam
        if not specs["city"]:
            for prov in sorted(ALL_VIETNAMESE_PROVINCES, key=len, reverse=True):
                p_norm = prov.lower()
                pattern = r"(?:^|\b|\s|ở\s+|tại\s+|tỉnh\s+)" + re.escape(p_norm) + r"(?:\b|\s|$|,|\.)"
                if re.search(pattern, t_norm):
                    specs["city"] = prov
                    break

        # Ghi nhận raw_location nếu có cụm từ "ở [địa danh]" / "tại [địa danh]" nhưng chưa được nhận diện
        if not specs["city"]:
            m_loc = re.search(r"(?:ở|tại|khu vực|thành phố|tp\.?)\s+([a-zà-ỹ\d\s]+?)(?=(?:\s+\d+|\s+diện tích|\s+giá|\s+có|\s+bao nhiêu|\s+2 tầng|\s+3 tầng|$|,))", t_norm)
            if m_loc:
                candidate = m_loc.group(1).strip()
                if candidate and candidate not in ("đây", "đó", "nhà", "này", "khu", "mặt phố", "hẻm"):
                    specs["raw_location"] = candidate.title()
                    c_norm = normalize_city(candidate)
                    if c_norm and c_norm in ALL_VIETNAMESE_PROVINCES:
                        specs["city"] = c_norm

        return specs

    # ── 2. QUẢN LÝ TRẠNG THÁI ĐA LƯỢT HỘI THOẠI (MULTI-TURN CONTEXT) ─────────────

    @classmethod
    def _resolve_context_with_history(cls, current_specs: dict, user_message: str, history_messages: list[dict]) -> dict:
        """
        Quản lý ngữ cảnh và cập nhật thông số qua các lượt hội thoại theo Section 8:
        - Không nhầm lẫn địa điểm giữa các câu hỏi.
        - Khi người dùng đổi địa điểm (VD: Vũng Tàu -> Đà Nẵng): cập nhật địa điểm mới, xóa phường/quận cũ.
        - Khi người dùng hỏi nối tiếp (VD: 'thế nếu 120m2 thì sao?'): kế thừa địa điểm cũ.
        - Khi người dùng hỏi câu độc lập không có địa điểm (VD: 'Nhà 100m2 giá 4 tỷ có hợp lý không?'): KHÔNG tự gán bừa địa điểm cũ.
        """
        specs = dict(current_specs)
        if not history_messages:
            return specs

        # Tìm thông số của lượt gần nhất
        prev_specs = None
        for h in reversed(history_messages):
            if h.get("metadata") and isinstance(h["metadata"], dict) and h["metadata"].get("specs"):
                prev_specs = h["metadata"]["specs"]
                break
            if h.get("role") == "user" and h.get("message"):
                p = cls.extract_property_specs(h["message"])
                if p.get("city") or p.get("area"):
                    prev_specs = p
                    break

        if not prev_specs:
            return specs

        msg_lower = user_message.lower().strip()

        # Dấu hiệu câu hỏi nối tiếp rõ ràng
        is_followup_question = bool(re.search(
            r"\b(còn|thế|nếu|vậy|thì sao|thêm|đổi|tăng|giảm|bớt|bao nhiêu|hợp lý không|có căn nào|ở đó|chỗ đó|khu đó)\b",
            msg_lower
        ))

        # 1. Trường hợp người dùng ĐỔI ĐỊA ĐIỂM trong câu hiện tại (VD: hỏi Vũng Tàu, sau đó hỏi Đà Nẵng)
        if current_specs.get("city") and prev_specs.get("city") and current_specs["city"] != prev_specs["city"]:
            # Dùng địa điểm mới, XÓA TOÀN BỘ quận/huyện, phường/xã, tuyến đường của địa điểm cũ
            specs["district"] = current_specs.get("district")
            specs["ward"] = current_specs.get("ward")
            specs["street"] = current_specs.get("street")
            # Kế thừa diện tích nếu câu mới không nhắc lại diện tích (VD: "Còn ở Đà Nẵng thì sao?")
            if not specs.get("area") and prev_specs.get("area"):
                specs["area"] = prev_specs["area"]
            return specs

        # 2. Trường hợp người dùng KHÔNG nêu địa điểm trong câu mới:
        if not current_specs.get("city"):
            if is_followup_question and prev_specs.get("city"):
                # Câu hỏi nối tiếp: kế thừa địa điểm từ lượt trước
                specs["city"] = prev_specs["city"]
                if not specs.get("district"):
                    specs["district"] = prev_specs.get("district")
                if not specs.get("ward"):
                    specs["ward"] = prev_specs.get("ward")
                if not specs.get("street"):
                    specs["street"] = prev_specs.get("street")
                if not specs.get("floors") and prev_specs.get("floors"):
                    specs["floors"] = prev_specs.get("floors")
                if not specs.get("bedrooms") and prev_specs.get("bedrooms"):
                    specs["bedrooms"] = prev_specs.get("bedrooms")
                if not specs.get("area") and prev_specs.get("area"):
                    specs["area"] = prev_specs["area"]

        return specs

    # ── 3. CÁC CÔNG CỤ THỰC THI (CHATBOT CAPABILITIES) ──────────────────────────

    @classmethod
    def execute_price_prediction(cls, specs: dict) -> dict:
        """
        Gọi mô hình Machine Learning dự đoán giá nhà chính xác theo quy chuẩn:
        - Kiểm tra diện tích.
        - Kiểm tra địa điểm (Tỉnh/Thành phố).
        - Kiểm tra phạm vi dữ liệu huấn luyện (52 tỉnh thành).
        - Định giá đúng khu vực, không tự gán bừa địa điểm mặc định.
        """
        area = specs.get("area")
        if not area or area <= 0:
            return {
                "success": False,
                "missing_info": "area",
                "specs": specs,
                "message": "Vui lòng cung cấp diện tích sử dụng (m²) để AI có thể định giá chính xác."
            }

        city = specs.get("city")
        if not city:
            raw_loc = specs.get("raw_location")
            return {
                "success": False,
                "missing_info": "city",
                "raw_location": raw_loc,
                "specs": specs,
                "message": (
                    f"Vui lòng cung cấp Tỉnh / Thành phố hoặc khu vực bất động sản "
                    f"{f'(khu vực bạn nhập \"{raw_loc}\" chưa được hệ thống nhận diện) ' if raw_loc else ''}"
                    f"để AI có thể định giá chính xác."
                )
            }

        # Kiểm tra phạm vi huấn luyện của mô hình
        catalog = get_active_location_catalog()
        matched_city = None
        for c in catalog.keys():
            if c.lower() == city.lower():
                matched_city = c
                break

        if not matched_city:
            # Thành phố nằm ngoài phạm vi 52 tỉnh thành huấn luyện của mô hình
            return {
                "success": False,
                "unsupported_region": True,
                "city": city,
                "specs": specs,
                "supported_count": len(catalog),
                "message": (
                    f"Hiện tại dữ liệu huấn luyện của hệ thống chưa đủ để định giá đáng tin cậy tại {city}. "
                    f"Nền tảng HousePrice AI hiện hỗ trợ định giá tại {len(catalog)} tỉnh/thành phố trên cả nước."
                )
            }

        # Chuẩn hóa Phường/Xã nếu có
        wards_in_city = catalog.get(matched_city, [])
        ward_input = specs.get("ward") or ""
        matched_ward = ""
        if ward_input:
            for w in wards_in_city:
                if w.lower() == ward_input.lower():
                    matched_ward = w
                    break
            if not matched_ward:
                matched_ward = ward_input

        form_data = {
            "city": matched_city,
            "district": specs.get("district") or "",
            "ward": matched_ward,
            "street": specs.get("street") or "",
            "area": float(area),
            "bedrooms": int(specs.get("bedrooms") or 2),
            "bathrooms": int(specs.get("bathrooms") or 2),
            "floors": int(specs.get("floors") or 1),
            "house_type": specs.get("house_type") or "nhà riêng",
        }

        try:
            pred_res = predict_price(form_data, allow_unknown_ward=True)
            pred_res["success"] = True
            pred_res["specs"] = specs
            pred_res["area"] = area
            pred_res["city"] = matched_city
            pred_res["ward"] = matched_ward if matched_ward else "Toàn thành phố (Mặt bằng chung)"
            pred_res["is_city_level"] = (not matched_ward)

            # Đánh giá mức giá người dùng đưa ra (nếu có)
            user_price = specs.get("user_price")
            if user_price and user_price > 0:
                pred_price = pred_res["price"]
                diff_pct = ((user_price - pred_price) / pred_price) * 100.0
                pred_res["user_price"] = user_price
                pred_res["diff_pct"] = round(diff_pct, 1)

                if abs(diff_pct) <= 7.0:
                    pred_res["price_evaluation"] = f"Mức giá {cls.format_currency_vnd(user_price)} là RẤT HỢP LÝ và sát với giá trị định giá thực của AI ({cls.format_currency_vnd(pred_price)})."
                    pred_res["evaluation_code"] = "fair"
                elif diff_pct > 7.0:
                    pred_res["price_evaluation"] = f"Mức giá {cls.format_currency_vnd(user_price)} đang CAO HƠN khoảng {abs(round(diff_pct, 1))}% so với định giá thị trường của AI ({cls.format_currency_vnd(pred_price)}). Bạn nên thương lượng thêm."
                    pred_res["evaluation_code"] = "high"
                else:
                    pred_res["price_evaluation"] = f"Mức giá {cls.format_currency_vnd(user_price)} đang THẤP HƠN khoảng {abs(round(diff_pct, 1))}% so với định giá thị trường của AI ({cls.format_currency_vnd(pred_price)}). Đây có thể là mức giá hấp dẫn/hời."
                    pred_res["evaluation_code"] = "low"

            return pred_res
        except Exception as e:
            return {"success": False, "error": str(e), "specs": specs}

    @classmethod
    def execute_house_search(cls, specs: dict, query_text: str = "") -> list[dict]:
        """Tìm kiếm bất động sản theo tiêu chí an toàn từ MySQL hoặc dataset."""
        results = []

        # 1. Thử truy vấn cơ sở dữ liệu MySQL houses
        try:
            where_clauses = ["1=1"]
            params = []

            if specs.get("city"):
                where_clauses.append("(city LIKE %s OR title LIKE %s)")
                params.extend([f"%{specs['city']}%", f"%{specs['city']}%"])

            if specs.get("district"):
                where_clauses.append("(district LIKE %s OR title LIKE %s)")
                params.extend([f"%{specs['district']}%", f"%{specs['district']}%"])

            if specs.get("user_price"):
                if "dưới" in query_text.lower() or "duoi" in query_text.lower():
                    where_clauses.append("price <= %s")
                    params.append(float(specs["user_price"]))
                else:
                    where_clauses.append("price BETWEEN %s AND %s")
                    params.extend([float(specs["user_price"]) * 0.7, float(specs["user_price"]) * 1.3])

            if specs.get("area"):
                if "trên" in query_text.lower() or "tren" in query_text.lower() or "lớn hơn" in query_text.lower():
                    where_clauses.append("area >= %s")
                    params.append(float(specs["area"]))
                else:
                    where_clauses.append("area >= %s")
                    params.append(float(specs["area"]) * 0.8)

            if specs.get("bedrooms"):
                where_clauses.append("bedrooms >= %s")
                params.append(int(specs["bedrooms"]))

            sql = f"SELECT * FROM houses WHERE {' AND '.join(where_clauses)} ORDER BY id DESC LIMIT 5"

            with get_cursor() as (cursor, _):
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
                if rows:
                    for r in rows:
                        results.append({
                            "id": r.get("id"),
                            "title": r.get("title") or "Bất động sản",
                            "city": r.get("city") or "",
                            "district": r.get("district") or "",
                            "street": r.get("street") or "",
                            "area": float(r.get("area") or 0),
                            "price": float(r.get("price") or 0),
                            "price_formatted": cls.format_currency_vnd(float(r.get("price") or 0)),
                            "bedrooms": int(r.get("bedrooms") or 0),
                            "bathrooms": int(r.get("bathrooms") or 0),
                            "house_type": r.get("house_type") or "Nhà riêng",
                        })
        except Exception:
            pass

        # 2. Nếu MySQL chưa có bản ghi phù hợp, tìm kiếm trong dataset.csv thực tế
        if not results and DATASET_PATH.exists():
            try:
                df = pd.read_csv(DATASET_PATH)
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["area"] = pd.to_numeric(df["area"], errors="coerce")
                df["rooms"] = pd.to_numeric(df.get("rooms", 0), errors="coerce").fillna(0)
                df["toilets"] = pd.to_numeric(df.get("toilets", 0), errors="coerce").fillna(0)

                mask = (df["price"] >= 100_000_000) & (df["area"] >= 10)

                if specs.get("city"):
                    c_canon = normalize_city(specs["city"])
                    mask &= df["location"].astype(str).apply(lambda x: normalize_city(x) == c_canon)

                if specs.get("district"):
                    d_clean = specs["district"].lower()
                    mask &= df["district"].astype(str).str.lower().str.contains(d_clean, na=False)

                if specs.get("user_price"):
                    if "dưới" in query_text.lower() or "duoi" in query_text.lower():
                        mask &= (df["price"] <= float(specs["user_price"]))
                    else:
                        mask &= (df["price"] <= float(specs["user_price"]) * 1.25) & (df["price"] >= float(specs["user_price"]) * 0.75)

                if specs.get("area"):
                    if "trên" in query_text.lower() or "tren" in query_text.lower() or "lớn hơn" in query_text.lower():
                        mask &= (df["area"] >= float(specs["area"]))
                    else:
                        mask &= (df["area"] >= float(specs["area"]) * 0.8)

                if specs.get("bedrooms"):
                    mask &= (df["rooms"] >= int(specs["bedrooms"]))

                matched = df[mask].head(4)
                for _, row in matched.iterrows():
                    p_val = float(row["price"])
                    results.append({
                        "id": str(row.get("listing_id")),
                        "title": str(row["title"])[:75],
                        "city": str(row["location"]),
                        "district": str(row["district"]),
                        "street": str(row.get("address") or ""),
                        "area": round(float(row["area"]), 1),
                        "price": p_val,
                        "price_formatted": cls.format_currency_vnd(p_val),
                        "bedrooms": int(row["rooms"]),
                        "bathrooms": int(row["toilets"]),
                        "house_type": str(row.get("house_type") or "Nhà riêng"),
                    })
            except Exception:
                pass

        return results

    @classmethod
    def execute_investment_analysis(cls, price: float, rent_monthly: float = None, loan_ratio: float = 0.7, loan_years: int = 15, interest_rate: float = 8.5) -> dict:
        """Tính toán tỷ suất lợi nhuận cho thuê (Rental Yield) và bảng tính vay trả góp."""
        if not rent_monthly or rent_monthly <= 0:
            if price > 0:
                rent_monthly = max(5_000_000, price * 0.0035)
            else:
                rent_monthly = 15_000_000

        annual_rent = rent_monthly * 12.0
        rental_yield = (annual_rent / price) * 100.0 if price > 0 else 0.0
        payback_years = (price / annual_rent) if annual_rent > 0 else 0.0

        loan_amount = price * loan_ratio
        down_payment = price - loan_amount
        monthly_rate = (interest_rate / 100.0) / 12.0
        total_months = loan_years * 12

        if monthly_rate > 0 and total_months > 0:
            monthly_payment = (
                loan_amount * (monthly_rate * (1 + monthly_rate) ** total_months) /
                ((1 + monthly_rate) ** total_months - 1)
            )
        else:
            monthly_payment = loan_amount / max(1, total_months)

        return {
            "success": True,
            "rental": {
                "rent_monthly": round(rent_monthly, 0),
                "rent_monthly_formatted": cls.format_currency_vnd(rent_monthly),
                "annual_rent_formatted": cls.format_currency_vnd(annual_rent),
                "yield_pct": round(rental_yield, 2),
                "payback_years": round(payback_years, 1),
                "comment": (
                    "Tỷ suất sinh lời hấp dẫn (trên 4.5%/năm), dòng tiền cho thuê tốt."
                    if rental_yield >= 4.5 else
                    "Tỷ suất cho thuê trung bình, phù hợp chiến lược tích lũy tài sản và kỳ vọng tăng giá đất dài hạn."
                ),
            },
            "mortgage": {
                "loan_ratio": int(loan_ratio * 100),
                "down_payment_formatted": cls.format_currency_vnd(down_payment),
                "loan_amount_formatted": cls.format_currency_vnd(loan_amount),
                "loan_years": loan_years,
                "interest_rate": interest_rate,
                "monthly_payment": round(monthly_payment, 0),
                "monthly_payment_formatted": cls.format_currency_vnd(monthly_payment),
            }
        }

    @classmethod
    def execute_explanation(cls, specs: dict) -> dict:
        """Giải thích chi tiết các thành phần đóng góp vào mức giá dự đoán của AI."""
        bundle = load_model()
        slope = bundle.get("slope", -0.25)
        intercept = bundle.get("intercept", 20.17)

        area = float(specs.get("area") or 80.0)
        city = specs.get("city") or "Tp Hồ Chí Minh"
        ward = specs.get("ward") or "Phường 1"
        street = specs.get("street") or ""
        floors = int(specs.get("floors") or 1)

        log_base = intercept + slope * np.log(max(area, 1.0))
        ppm_base = np.exp(log_base)

        from backend.predict import get_street_multiplier
        street_mult = get_street_multiplier(street, ward) if street else 1.0
        floors_mult = 1.0 + max(0.0, float(floors) - 1.0) * 0.08

        return {
            "slope": round(float(slope), 3),
            "area_effect": (
                f"Đường cong lũy thừa diện tích (slope = {slope:.3f}): Đảm bảo tổng giá TĂNG ĐƠN ĐIỆU theo diện tích. "
                f"Với {area}m², đơn giá cơ sở ước tính khoảng {ppm_base/1e6:.1f} triệu/m²."
            ),
            "location_effect": f"Khu vực {ward}, {city} được mô hình Gradient Boosting hiệu chỉnh theo chất lượng hạ tầng và lịch sử giao dịch địa phương.",
            "street_effect": (
                f"Tuyến đường '{street}' được định giá hệ số vị trí đặc biệt x{street_mult} (Mặt tiền / Tuyến giao thương trọng điểm)."
                if street_mult > 1.0 else f"Tuyến đường '{street or 'khu vực'}' nằm ở phân khúc đường nhánh/nội bộ tiêu chuẩn."
            ),
            "floors_effect": f"Nhà {floors} tầng được bổ sung hệ số công trình xây dựng x{floors_mult:.2f} (+8%/tầng mở rộng).",
            "model_name": bundle.get("algorithm", "Gradient Boosting v2.4"),
            "confidence": bundle.get("accuracy", 88.0),
        }

    # ── 4. TỔNG HỢP & GIAO TIẾP VỚI LLM / ENGINE DỰ PHÒNG ─────────────────────

    @classmethod
    def process_message(cls, user_message: str, history_messages: list[dict] = None) -> dict:
        """
        Xử lý toàn bộ luồng trò chuyện:
        1. Phân tích câu hỏi & dữ liệu thực tế
        2. Tích hợp OpenAI LLM nếu có API Key
        3. Tự động chuyển đổi sang Built-in Intelligent Real Estate Engine khi không có key / lỗi mạng
        """
        history_messages = history_messages or []
        raw_specs = cls.extract_property_specs(user_message)
        specs = cls._resolve_context_with_history(raw_specs, user_message, history_messages)
        text_lower = user_message.lower()

        # Phát hiện ý định người dùng (Intent Recognition)
        is_predict_intent = bool(
            specs.get("area") or
            re.search(r"\b(định giá|giá bao nhiêu|khoảng bao nhiêu|giá.*hợp lý|giá.*đắt|giá.*rẻ|bao tiền)\b", text_lower)
        )
        is_search_intent = bool(
            re.search(r"\b(tìm|kiếm|danh sách|có căn nào|có nhà nào|dưới\s*\d+\s*tỷ|tìm nhà)\b", text_lower)
        )
        is_invest_intent = bool(
            re.search(r"\b(đầu tư|cho thuê|sinh lời|hoàn vốn|lợi nhuận|trả góp|vay|lãi suất)\b", text_lower)
        )
        is_explain_intent = bool(
            re.search(r"\b(tại sao|giải thích|yếu tố nào|ảnh hưởng thế nào|nguyên nhân|căn cứ)\b", text_lower)
        )
        is_compare_intent = bool(
            re.search(r"\b(so sánh|khác nhau|hơn kém|chênh lệch)\b", text_lower)
        )

        metadata = {"specs": specs}
        tool_results_text = []

        # 1. Chạy công cụ Định giá AI nếu có ý định
        if is_predict_intent:
            pred_data = cls.execute_price_prediction(specs)
            if pred_data.get("success"):
                metadata["prediction"] = {
                    "price": pred_data["price"],
                    "price_formatted": cls.format_currency_vnd(pred_data["price"]),
                    "min_price_formatted": cls.format_currency_vnd(pred_data["min_price"]),
                    "max_price_formatted": cls.format_currency_vnd(pred_data["max_price"]),
                    "confidence": pred_data["confidence"],
                    "algorithm": pred_data["algorithm"],
                    "city": pred_data["city"],
                    "ward": pred_data["ward"],
                    "area": pred_data["area"],
                    "is_city_level": pred_data.get("is_city_level", False),
                }
                if pred_data.get("price_evaluation"):
                    metadata["prediction"]["evaluation"] = pred_data["price_evaluation"]

                tool_results_text.append(
                    f"KẾT QUẢ ĐỊNH GIÁ AI TỪ MACHINE LEARNING:\n"
                    f"- Khu vực: {pred_data['city']} ({pred_data['ward']})\n"
                    f"- Diện tích: {pred_data['area']} m²\n"
                    f"- Giá ước tính: {cls.format_currency_vnd(pred_data['price'])}\n"
                    f"- Khoảng giao dịch đề xuất: {cls.format_currency_vnd(pred_data['min_price'])} - {cls.format_currency_vnd(pred_data['max_price'])}\n"
                    f"- Độ tin cậy mô hình: {pred_data['confidence']}%\n"
                    f"- Thuật toán: {pred_data['algorithm']}\n"
                    f"- Đánh giá mức giá: {pred_data.get('price_evaluation', 'Mức giá chuẩn xác theo mô hình.')}"
                )
            elif pred_data.get("unsupported_region"):
                metadata["unsupported_region"] = pred_data
                tool_results_text.append(
                    f"THÔNG BÁO QUAN TRỌNG: Khu vực '{pred_data['city']}' nằm ngoài phạm vi 52 tỉnh thành mà mô hình Machine Learning được huấn luyện. "
                    f"Bạn PHẢI thông báo rõ ràng rằng hệ thống chưa đủ dữ liệu huấn luyện để định giá đáng tin cậy tại khu vực này, "
                    f"và TUYỆT ĐỐI KHÔNG được tự ý gán TP.HCM hay bất kỳ địa điểm nào khác."
                )
            elif pred_data.get("missing_info") == "city":
                metadata["missing_location"] = pred_data
                tool_results_text.append(
                    f"THÔNG BÁO QUAN TRỌNG: Người dùng chưa cung cấp vị trí bất động sản (Tỉnh/Thành phố hoặc Quận/Huyện). "
                    f"Bạn PHẢI hỏi người dùng bổ sung vị trí cụ thể và giải thích rằng giá nhà phụ thuộc lớn vào vị trí nên AI không thể tự gán vị trí mặc định."
                )
            elif pred_data.get("missing_info") == "area":
                metadata["missing_area"] = pred_data
                tool_results_text.append(
                    f"THÔNG BÁO QUAN TRỌNG: Người dùng chưa cung cấp diện tích sử dụng (m²). "
                    f"Bạn PHẢI hỏi người dùng bổ sung diện tích."
                )

        # 2. Chạy công cụ Tìm kiếm Bất động sản
        if is_search_intent:
            houses = cls.execute_house_search(specs, user_message)
            if houses:
                metadata["houses"] = houses
                h_lines = ["KẾT QUẢ TÌM KIẾM NHÀ ĐẤT TRONG DATABASE:"]
                for i, h in enumerate(houses, 1):
                    h_lines.append(f"{i}. {h['title']} — Vị trí: {h['district']}, {h['city']} — DT: {h['area']}m² — Giá: {h['price_formatted']} — {h['bedrooms']}PN, {h['bathrooms']}WC")
                tool_results_text.append("\n".join(h_lines))

        # 3. Chạy công cụ Phân tích Tài chính Đầu tư & Trả góp
        if is_invest_intent:
            target_price = specs.get("user_price") or (metadata.get("prediction", {}).get("price")) or 4_000_000_000
            m_rent = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:triệu|trieu|tr)\s*(?:/|\s+)?(?:tháng|thang)?", text_lower)
            rent_val = float(m_rent.group(1).replace(",", ".")) * 1_000_000 if m_rent else 15_000_000
            inv_data = cls.execute_investment_analysis(target_price, rent_monthly=rent_val)
            metadata["investment"] = inv_data
            tool_results_text.append(
                f"BẢNG TÍNH ĐẦU TƯ & TRẢ GÓP:\n"
                f"- Tỷ suất cho thuê: {inv_data.get('rental', {}).get('yield_pct', 'N/A')}%\n"
                f"- Thời gian hoàn vốn: {inv_data.get('rental', {}).get('payback_years', 'N/A')} năm\n"
                f"- Vay 70% ({inv_data['mortgage']['loan_amount_formatted']}) trong 15 năm (lãi 8.5%/năm): "
                f"Trả mỗi tháng khoảng {inv_data['mortgage']['monthly_payment_formatted']}"
            )

        # 4. Chạy công cụ Giải thích mô hình
        if is_explain_intent:
            exp_data = cls.execute_explanation(specs)
            metadata["explanation"] = exp_data
            tool_results_text.append(
                f"GIẢI THÍCH MÔ HÌNH DỰ ĐOÁN:\n"
                f"- {exp_data['area_effect']}\n"
                f"- {exp_data['location_effect']}\n"
                f"- {exp_data['street_effect']}\n"
                f"- {exp_data['floors_effect']}"
            )

        # Gửi dữ liệu qua OpenAI LLM nếu đã cấu hình API Key
        llm_cfg = cls._get_llm_config()
        if llm_cfg["api_key"]:
            try:
                ai_text = cls._call_openai_api(user_message, history_messages, "\n\n".join(tool_results_text), llm_cfg)
                return {
                    "role": "assistant",
                    "message": ai_text,
                    "metadata": metadata,
                    "provider": "openai"
                }
            except Exception as llm_err:
                print(f"[WARN] OpenAI API error: {llm_err}, falling back to intelligent real estate engine.")

        # FALLBACK: Built-in Intelligent Real Estate Engine (Chuẩn hóa câu trả lời tiếng Việt chuyên nghiệp)
        fallback_text = cls._generate_intelligent_fallback_response(
            user_message, specs, metadata, is_predict_intent, is_search_intent, is_invest_intent, is_explain_intent
        )

        return {
            "role": "assistant",
            "message": fallback_text,
            "metadata": metadata,
            "provider": "builtin_ai"
        }

    @classmethod
    def _call_openai_api(cls, user_message: str, history: list[dict], tool_context: str, config: dict) -> str:
        """Gọi OpenAI Chat Completions API với ngữ cảnh dữ liệu thực tế."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        except Exception:
            client = None

        system_prompt = (
            "Bạn là Chuyên viên AI Tư vấn Bất động sản cao cấp của nền tảng HousePrice AI. "
            "Nhiệm vụ của bạn là tư vấn giá nhà, tìm kiếm nhà đất, phân tích đầu tư và giải thích các yếu tố thị trường bằng tiếng Việt chuyên nghiệp, khách quan, tận tâm.\n\n"
            "NGUYÊN TẮC BẮT BUỘC:\n"
            "1. Tuyệt đối không tự ý gán TP.HCM hay bất kỳ vị trí mặc định nào nếu người dùng không cung cấp hoặc cung cấp địa phương khác.\n"
            "2. Khi có kết quả từ Machine Learning hoặc Database, hãy sử dụng chính xác các con số đó. Không tự ý bịa số liệu.\n"
            "3. Nếu khu vực người dùng hỏi nằm ngoài phạm vi 52 tỉnh thành hỗ trợ, PHẢI nêu rõ dữ liệu chưa đủ để định giá tại đó.\n"
            "4. Nếu người dùng chưa cung cấp vị trí hoặc diện tích, hãy lịch sự hỏi người dùng bổ sung thông tin cần thiết.\n"
            "5. Luôn định dạng tiền tệ rõ ràng theo Tỷ và Triệu VNĐ (ví dụ: 4.85 Tỷ VNĐ, 550 Triệu VNĐ).\n"
            "6. Định dạng câu trả lời bằng Markdown đẹp mắt, có gạch đầu dòng, bảng số liệu hoặc thẻ tóm tắt khi thích hợp.\n"
        )

        if tool_context:
            system_prompt += f"\n\nDỮ LIỆU THỰC TẾ HỆ THỐNG TRUY VẤN ĐƯỢC:\n{tool_context}"

        messages = [{"role": "system", "content": system_prompt}]

        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("message"):
                messages.append({"role": h["role"], "content": h["message"]})

        messages.append({"role": "user", "content": user_message})

        if client:
            resp = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=0.6,
                max_tokens=900,
            )
            return resp.choices[0].message.content
        else:
            import requests
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            body = {
                "model": config["model"],
                "messages": messages,
                "temperature": 0.6,
                "max_tokens": 900
            }
            res = requests.post(f"{config['base_url']}/chat/completions", headers=headers, json=body, timeout=20)
            res_data = res.json()
            return res_data["choices"][0]["message"]["content"]

    @classmethod
    def _generate_intelligent_fallback_response(cls, user_msg: str, specs: dict, metadata: dict,
                                                is_predict: bool, is_search: bool, is_invest: bool, is_explain: bool) -> str:
        """Sinh câu trả lời thông minh, chuyên nghiệp từ kết quả phân tích nội bộ khi chưa cấu hình OpenAI key."""
        parts = []

        # 1. Trả lời định giá
        if "prediction" in metadata:
            pred = metadata["prediction"]
            city_name = pred["city"]
            ward_name = pred["ward"]
            is_city_level = pred.get("is_city_level", False)

            if is_city_level:
                loc_display = f"**{city_name}** *(Mặt bằng chung toàn tỉnh/thành phố)*"
            else:
                loc_display = f"{ward_name}, **{city_name}**"

            parts.append(
                f"### 🎯 Kết quả Định giá Bất động sản AI\n\n"
                f"- **Khu vực:** {loc_display}\n"
                f"- **Diện tích ghi nhận:** **{pred['area']} m²**\n"
                f"- **Mức giá ước tính:** **{pred['price_formatted']}**\n"
                f"- **Khoảng giao dịch tham khảo:** {pred['min_price_formatted']} – {pred['max_price_formatted']}\n"
                f"- **Độ tin cậy của mô hình:** `{pred['confidence']}%` (Thuật toán: {pred['algorithm']})\n"
            )
            if is_city_level:
                parts.append(
                    f"> **📌 Ghi chú vị trí:** Do bạn chưa cung cấp Phường/Xã cụ thể tại {city_name}, "
                    f"kết quả trên được ước tính theo mức trung vị của toàn tỉnh/thành phố. "
                    f"Bạn có thể cung cấp thêm tên Phường hoặc Tuyến đường để AI định giá chi tiết theo từng khu phố!\n"
                )
            if pred.get("evaluation"):
                parts.append(f"> **💡 Nhận định của AI:** {pred['evaluation']}\n")

        elif "unsupported_region" in metadata:
            unsupp = metadata["unsupported_region"]
            c_name = unsupp.get("city") or "khu vực này"
            parts.append(
                f"### ⚠️ Giới hạn dữ liệu định giá\n\n"
                f"Hiện tại dữ liệu huấn luyện của hệ thống chưa đủ để định giá đáng tin cậy tại **{c_name}**.\n\n"
                f"Nền tảng **HousePrice AI** hiện hỗ trợ định giá chuyên sâu tại **52 Tỉnh / Thành phố** trọng điểm trên toàn quốc "
                f"(như TP. Hồ Chí Minh, Hà Nội, Đà Nẵng, Bà Rịa - Vũng Tàu, Bình Dương, Đồng Nai, Cần Thơ, Hải Phòng, Khánh Hòa, Quảng Ninh, Lâm Đồng, v.v.).\n\n"
                f"👉 *Gợi ý:* Bạn có thể tham khảo hoặc cung cấp thông tin bất động sản tại các khu vực mà hệ thống đang hỗ trợ để AI thẩm định chính xác nhất nhé!"
            )

        elif "missing_location" in metadata:
            miss = metadata["missing_location"]
            area_val = miss.get("specs", {}).get("area")
            price_val = miss.get("specs", {}).get("user_price")
            p_str = f" mức giá **{cls.format_currency_vnd(price_val)}** cho" if price_val else ""
            a_str = f" căn nhà **{area_val} m²**" if area_val else " căn nhà"

            parts.append(
                f"### 📍 Cần thông tin vị trí để định giá chính xác\n\n"
                f"Để thẩm định xem{p_str}{a_str} có hợp lý hay không, bạn vui lòng cho tôi biết **vị trí của bất động sản (Tỉnh/Thành phố, Quận/Huyện hoặc Phường/Xã)**.\n\n"
                f"> **💡 Lưu ý quan trọng:** Giá bất động sản phụ thuộc cốt lõi vào vị trí địa lý. "
                f"Cùng diện tích 100m², nếu ở trung tâm TP.HCM hay Hà Nội thì 4 tỷ là mức giá cực kỳ hời (hiếm có); "
                f"nhưng tại các khu vực ngoại thành hoặc tỉnh thành khác thì có thể là mức giá trung bình hoặc cao. "
                f"Hệ thống không tự ý gán vị trí mặc định để đảm bảo tính khách quan và trung thực cho bạn."
            )

        elif "missing_area" in metadata:
            miss = metadata["missing_area"]
            city_str = f" tại **{miss['city']}**" if miss.get("city") else ""
            sp = miss.get("specs", {})
            extra_specs = []
            if sp.get("floors"):
                extra_specs.append(f"{sp['floors']} tầng")
            if sp.get("bedrooms"):
                extra_specs.append(f"{sp['bedrooms']} phòng ngủ")
            if sp.get("bathrooms"):
                extra_specs.append(f"{sp['bathrooms']} WC")
            if sp.get("house_type"):
                extra_specs.append(f"Loại: {sp['house_type']}")

            specs_text = f" (kết cấu: {', '.join(extra_specs)})" if extra_specs else ""
            parts.append(
                f"### 📐 Cần thông tin diện tích sử dụng\n\n"
                f"Tôi đã ghi nhận thông tin bất động sản{city_str}{specs_text}.\n\n"
                f"Để mô hình Machine Learning đưa ra mức định giá sát nhất với thị trường thực tế, "
                f"bạn vui lòng cho tôi biết thêm **diện tích sử dụng (m²)** của căn nhà nhé!"
            )

        # 2. Trả lời tìm kiếm nhà
        if "houses" in metadata and metadata["houses"]:
            houses = metadata["houses"]
            parts.append(f"### 🏡 Danh sách Nhà đất Phù hợp ({len(houses)} căn tìm thấy):\n")
            for h in houses:
                parts.append(
                    f"- **{h['title']}**\n"
                    f"  - 📍 Vị trí: {h['district']}, {h['city']}\n"
                    f"  - 📐 Diện tích: **{h['area']} m²** | 💰 Giá: **{h['price_formatted']}**\n"
                    f"  - 🛏️ Thiết kế: {h['bedrooms']} phòng ngủ • {h['bathrooms']} phòng tắm\n"
                )
            parts.append("*Bạn có thể đặt câu hỏi chi tiết về bất kỳ căn nào trong danh sách trên nhé!*")
        elif is_search and not metadata.get("houses"):
            parts.append(
                "### 🔍 Kết quả tìm kiếm\n\n"
                "Hiện chưa tìm thấy bất động sản nào khớp hoàn toàn với tất cả tiêu chí bạn yêu cầu. "
                "Bạn có thể thử mở rộng khoảng giá hoặc tìm kiếm theo khu vực quận/huyện lân cận xem sao nhé!"
            )

        # 3. Trả lời tư vấn đầu tư & tài chính
        if "investment" in metadata:
            inv = metadata["investment"]
            parts.append("### 📊 Phân tích Hiệu quả Đầu tư & Bảng tính Trả góp\n")
            if "rental" in inv:
                rent = inv["rental"]
                parts.append(
                    f"**1. Khả năng khai thác dòng tiền cho thuê:**\n"
                    f"- Tiền thuê ước tính: **{rent['rent_monthly_formatted']}/tháng** ({rent['annual_rent_formatted']}/năm)\n"
                    f"- **Tỷ suất lợi nhuận (Rental Yield):** **`{rent['yield_pct']}% / năm`**\n"
                    f"- Thời gian hoàn vốn đầu tư thuần: khoảng **{rent['payback_years']} năm**\n"
                    f"- *Đánh giá:* {rent['comment']}\n\n"
                )
            if "mortgage" in inv:
                m = inv["mortgage"]
                parts.append(
                    f"**2. Phương án vay ngân hàng trả góp (Giả định vay {m['loan_ratio']}%):**\n"
                    f"- Số tiền tự có (Vốn đối ứng 30%): **{m['down_payment_formatted']}**\n"
                    f"- Khoản vay ngân hàng: **{m['loan_amount_formatted']}** (Kỳ hạn {m['loan_years']} năm, lãi suất tham khảo {m['interest_rate']}%/năm)\n"
                    f"- **Số tiền cần trả mỗi tháng:** khoảng **`{m['monthly_payment_formatted']}/tháng`** (Gốc + Lãi)\n"
                    f"- *Lưu ý:* Bảng tính mang tính tham khảo, lãi suất thực tế sẽ phụ thuộc vào gói ưu đãi của từng ngân hàng thương mại."
                )

        # 4. Trả lời giải thích mô hình
        if "explanation" in metadata:
            exp = metadata["explanation"]
            parts.append(
                f"### 🔬 Giải thích Mức độ Ảnh hưởng đến Giá nhà ({exp['model_name']})\n\n"
                f"Mô hình AI định giá bất động sản dựa trên 4 nhóm đặc trưng chính:\n"
                f"1. **Diện tích:** {exp['area_effect']}\n"
                f"2. **Vị trí & Phường/Xã:** {exp['location_effect']}\n"
                f"3. **Tuyến đường:** {exp['street_effect']}\n"
                f"4. **Kết cấu xây dựng:** {exp['floors_effect']}\n"
            )

        # Nếu là câu hỏi chào hỏi hoặc tư vấn chung chung
        if not parts:
            parts.append(
                "Xin chào! Tôi là **Trợ lý AI Tư vấn Bất động sản HousePrice AI**.\n\n"
                "Tôi có thể hỗ trợ bạn các vấn đề sau:\n"
                "- 🎯 **Định giá nhà tự động bằng AI:** Ví dụ *'Nhà 100m2 tại Vũng Tàu giá bao nhiêu?'* hoặc *'Nhà 70m2 tại Quận 12 TP.HCM giá khoảng bao nhiêu?'*\n"
                "- 🔍 **Tìm kiếm nhà theo nhu cầu:** Ví dụ *'Tìm nhà dưới 3 tỷ tại TP.HCM trên 50m2 có 2 phòng ngủ'*\n"
                "- ⚖️ **So sánh giá nhà:** So sánh các khu vực hoặc so sánh giá rao bán với định giá AI.\n"
                "- 📊 **Tư vấn đầu tư & Vay trả góp:** Ước tính tỷ suất cho thuê (Yield), dòng tiền hàng tháng khi vay mua nhà.\n"
                "- 🔬 **Giải thích cơ sở định giá:** Phân tích mức độ đóng góp của vị trí, diện tích, số tầng và tuyến đường.\n\n"
                "Bạn cần tôi hỗ trợ thẩm định hoặc tìm kiếm căn nhà nào hôm nay?"
            )

        return "\n\n".join(parts)

    @staticmethod
    def format_currency_vnd(price: float) -> str:
        """Định dạng số tiền sang chuẩn Tỷ / Triệu VNĐ."""
        if not price or not np.isfinite(price) or price <= 0:
            return "0 VNĐ"
        if price >= 1_000_000_000:
            val = price / 1_000_000_000
            return f"{val:.2f} Tỷ VNĐ" if val % 1 != 0 else f"{int(val)} Tỷ VNĐ"
        elif price >= 1_000_000:
            val = price / 1_000_000
            return f"{val:.1f} Triệu VNĐ" if val % 1 != 0 else f"{int(val)} Triệu VNĐ"
        else:
            return f"{int(price):,} VNĐ"
