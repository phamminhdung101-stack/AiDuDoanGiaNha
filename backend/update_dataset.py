"""Tải và chuẩn hóa dataset nhà bán TP.HCM từ Hugging Face.

Chạy từ thư mục gốc dự án:
    python -m backend.update_dataset

Tải theo streaming để không phải nạp toàn bộ dataset 3.5 triệu dòng vào RAM.
Dataset nguồn có giấy phép CC BY-NC 4.0: chỉ dùng cho học tập/nghiên cứu phi thương mại.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "database" / "dataset.csv"
DATASET_ID = "tinixai/vietnam-real-estates"
SOURCE_NAME = "Tinix AI - Vietnam Real Estate Listings (2025-2026)"

# Chỉ lấy nhóm nhà ở, không trộn đất nền hoặc căn hộ chung cư vào mô hình nhà.
ALLOWED_TYPES = {
    "nhà", "nha", "biệt thự/nhà liền kề", "biet thu/nha lien ke",
    "biệt thự / nhà liền kề", "biet thu / nha lien ke",
}
RENT_PATTERN = re.compile(
    r"cho thuê|cần thuê|tìm thuê|thuê nhà|thuê nguyên căn|"
    r"\bcho thue\b|\bcan thue\b|\btim thue\b|\bthue nha\b|"
    r"/tháng|/thang|mỗi tháng|moi thang|per month", re.IGNORECASE,
)
SALE_PATTERN = re.compile(
    r"bán|cần bán|chuyển nhượng|sang nhượng|ban nha|can ban|chuyen nhuong",
    re.IGNORECASE,
)

FIELDS = [
    "title", "description", "property_type_name", "price", "area",
    "location", "ward", "rooms", "toilets", "address",
    "price_per_m2", "published_at", "district_name", "street_name",
    "floor_count", "frontage_width", "road_width", "house_direction",
]


def is_hcm(value: object) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.search(r"hồ\s*chí\s*minh|ho\s*chi\s*minh|tp\.?\s*hcm|tphcm", text))


def as_number(value: object) -> float | None:
    try:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Cập nhật dataset nhà bán TP.HCM")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Giới hạn số dòng ghi ra để thử nghiệm; 0 = không giới hạn")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Thiếu thư viện datasets. Hãy chạy: pip install 'datasets>=3.0.0'"
        ) from exc

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".dataset_update_", suffix=".csv",
                                     dir=OUTPUT_PATH.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    scanned = kept = 0
    try:
        print(f"Đang đọc dataset {DATASET_ID} theo streaming...")
        stream = load_dataset(DATASET_ID, split="train", streaming=True)
        with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in stream:
                scanned += 1
                if not is_hcm(row.get("province_name")):
                    continue
                prop_type = str(row.get("property_type_name") or "").strip()
                if prop_type.lower() not in ALLOWED_TYPES:
                    continue
                title = str(row.get("name") or "")
                description = str(row.get("description") or "")
                all_text = title + " " + description
                # Tiêu đề cho thuê bị loại; tin bán vẫn giữ nếu mô tả nhắc nhà đang có hợp đồng thuê.
                if RENT_PATTERN.search(title) or (RENT_PATTERN.search(all_text) and not SALE_PATTERN.search(all_text)):
                    continue

                price = as_number(row.get("price"))
                area = as_number(row.get("area"))
                if price is None or area is None:
                    continue
                if not (100_000_000 <= price <= 200_000_000_000 and 10 <= area <= 1_000):
                    continue
                ppm = price / area
                if not (5_000_000 <= ppm <= 1_000_000_000):
                    continue

                ward = str(row.get("ward_name") or "").strip()
                district = str(row.get("district_name") or "").strip()
                street = str(row.get("street_name") or "").strip()
                address = ", ".join(part for part in (street, ward, district, "TP. Hồ Chí Minh") if part)
                writer.writerow({
                    "title": title,
                    "description": description,
                    "property_type_name": prop_type,
                    "price": price,
                    "area": area,
                    "location": "Tp Hồ Chí Minh",
                    "ward": ward,
                    "rooms": row.get("bedroom_count"),
                    "toilets": row.get("bathroom_count"),
                    "address": address,
                    "price_per_m2": ppm,
                    "published_at": row.get("published_at"),
                    "district_name": district,
                    "street_name": street,
                    "floor_count": row.get("floor_count"),
                    "frontage_width": row.get("frontage_width"),
                    "road_width": row.get("road_width"),
                    "house_direction": row.get("house_direction"),
                })
                kept += 1
                if kept and kept % 10_000 == 0:
                    print(f"Đã giữ {kept:,} tin phù hợp (đã quét {scanned:,})...")
                if args.max_rows and kept >= args.max_rows:
                    break

        if kept < 50:
            raise RuntimeError(
                f"Chỉ lọc được {kept} tin; không thay thế dataset hiện tại để tránh mất dữ liệu."
            )
        # Atomic replace: nếu tải/lọc lỗi, dataset cũ vẫn được giữ nguyên.
        os.replace(temp_path, OUTPUT_PATH)
        print("\nCẬP NHẬT DATASET THÀNH CÔNG")
        print(f"Nguồn: {SOURCE_NAME}")
        print(f"Đã quét: {scanned:,} dòng | Giữ lại: {kept:,} dòng")
        print(f"Đã lưu: {OUTPUT_PATH}")
        print("Lưu ý giấy phép: CC BY-NC 4.0, chỉ sử dụng phi thương mại nếu chưa có quyền khác.")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
