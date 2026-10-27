"""
test_location_validation.py — Kiểm thử danh mục địa điểm, chuẩn hóa, xác thực dự đoán và an toàn cập nhật.
"""
import copy
import html
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import app
from backend.location_service import (
    normalize_city,
    normalize_ward,
    normalize_street,
    standardize_vietnamese_accents,
    build_location_catalog,
    validate_location,
    StreetValidator,
    get_street_data_assessment,
    get_active_location_catalog,
    get_active_street_catalog,
    load_street_catalog,
    clear_street_catalog_cache,
    save_location_catalog,
    CATALOG_PATH,
    STREET_CATALOG_PATH,
)
from backend.predict import predict_price, load_model, reload_model
from backend.train_model import clean_dataset, train, save_model_bundle, MODEL_PATH


class TestLocationCatalogGeneration(unittest.TestCase):
    """1. Kiểm thử tạo danh mục từ dataset & chuẩn hóa địa danh."""

    def test_catalog_generated_from_dataset(self):
        """Danh mục được tạo thành công từ dữ liệu sạch, giữ đúng quan hệ city -> wards."""
        df_sample = pd.DataFrame([
            {"location": "Tp Hồ Chí Minh", "ward": "Phường 1", "price": 3_000_000_000, "area": 50},
            {"location": "Tp Hồ Chí Minh", "ward": "Phường 2", "price": 4_000_000_000, "area": 60},
            {"location": "Hà Nội", "ward": "Phường Tràng Tiền", "price": 5_000_000_000, "area": 70},
            {"location": "Hà Nội", "ward": "Phường Hàng Bài", "price": 6_000_000_000, "area": 80},
        ])
        catalog = build_location_catalog(df_sample)
        self.assertIn("Tp Hồ Chí Minh", catalog)
        self.assertIn("Hà Nội", catalog)
        self.assertEqual(catalog["Tp Hồ Chí Minh"], ["Phường 1", "Phường 2"])
        self.assertEqual(catalog["Hà Nội"], ["Phường Hàng Bài", "Phường Tràng Tiền"])
        # Đảm bảo không gộp phường Hà Nội vào TP.HCM
        self.assertNotIn("Phường Tràng Tiền", catalog["Tp Hồ Chí Minh"])
        self.assertNotIn("Phường 1", catalog["Hà Nội"])

    def test_whitespace_and_empty_normalization(self):
        """Chuẩn hóa khoảng trắng thừa, Unicode và loại bỏ giá trị rỗng/unknown."""
        df_dirty = pd.DataFrame([
            {"location": "  Tp   Hồ  Chí   Minh  ", "ward": "  Phường  1  "},
            {"location": "Tp Hồ Chí Minh", "ward": "unknown"},
            {"location": "", "ward": "Phường 2"},
            {"location": "Hà Nội", "ward": "None"},
            {"location": "Hà Nội", "ward": "nan"},
            {"location": "Hà Nội", "ward": "  Phường   Hàng Bài  "},
        ])
        catalog = build_location_catalog(df_dirty)
        self.assertEqual(catalog["Tp Hồ Chí Minh"], ["Phường 1"])
        self.assertEqual(catalog["Hà Nội"], ["Phường Hàng Bài"])

    def test_accent_standardization_and_distinct_wards(self):
        """Chuẩn hóa vị trí dấu nhưng không tự ý gộp các địa danh khác nhau."""
        # 'Bình Hưng Hoà A' vs 'Bình Hưng Hòa' là 2 địa danh khác biệt
        w1 = normalize_ward("Phường Bình Hưng Hoà A")
        w2 = normalize_ward("Phường Bình Hưng Hòa")
        self.assertNotEqual(w1, w2)
        self.assertEqual(w1, "Phường Bình Hưng Hòa A")
        self.assertEqual(w2, "Phường Bình Hưng Hòa")

    def test_empty_dataset_raises_clear_error_without_hardcode(self):
        """Nếu dataset sau làm sạch không có địa điểm hợp lệ, báo lỗi rõ ràng, không hard-code."""
        df_invalid = pd.DataFrame([
            {"location": "unknown", "ward": "unknown", "price": 1000, "area": 50},
            {"location": "", "ward": "", "price": 2000, "area": 60},
        ])
        with self.assertRaises(ValueError) as ctx:
            build_location_catalog(df_invalid)
        self.assertIn("không chứa địa điểm", str(ctx.exception))


class TestPredictionValidation(unittest.TestCase):
    """2. Kiểm tra tính hợp lệ khi dự đoán ở backend."""

    def setUp(self):
        self.catalog = {
            "Tp Hồ Chí Minh": ["Phường 1", "Phường 2", "Phường Bến Nghé"],
            "Hà Nội": ["Phường Tràng Tiền", "Phường Hàng Bài"],
        }

    def test_valid_city_and_ward_accepted(self):
        """Phường đúng thành phố được chấp nhận."""
        city, ward = validate_location("Tp Hồ Chí Minh", "Phường 1", self.catalog)
        self.assertEqual(city, "Tp Hồ Chí Minh")
        self.assertEqual(ward, "Phường 1")

    def test_city_alias_accepted(self):
        """Bí danh 'Hồ Chí Minh' hoặc 'TP. Hồ Chí Minh' được chuẩn hóa về tên trong danh mục."""
        city, ward = validate_location("Hồ Chí Minh", "Phường 1", self.catalog)
        self.assertEqual(city, "Tp Hồ Chí Minh")
        self.assertEqual(ward, "Phường 1")

    def test_ward_of_different_city_rejected(self):
        """Phường thuộc thành phố khác bị từ chối với thông báo rõ ràng."""
        with self.assertRaises(ValueError) as ctx:
            validate_location("Tp Hồ Chí Minh", "Phường Tràng Tiền", self.catalog)
        msg = str(ctx.exception)
        self.assertIn("Hà Nội", msg)
        self.assertIn("không thuộc thành phố 'Tp Hồ Chí Minh'", msg)

    def test_nonexistent_city_rejected(self):
        """Thành phố không tồn tại bị từ chối với thông báo tiếng Việt."""
        with self.assertRaises(ValueError) as ctx:
            validate_location("Hải Phòng", "Phường 1", self.catalog)
        self.assertIn("không tồn tại trong danh mục", str(ctx.exception))

    def test_nonexistent_ward_rejected(self):
        """Phường không tồn tại bị từ chối với thông báo tiếng Việt."""
        with self.assertRaises(ValueError) as ctx:
            validate_location("Tp Hồ Chí Minh", "Phường 999", self.catalog)
        self.assertIn("không tồn tại trong danh mục của thành phố 'Tp Hồ Chí Minh'", str(ctx.exception))

    def test_invalid_area_rejected(self):
        """Diện tích <= 0 bị từ chối."""
        with self.assertRaises(ValueError) as ctx:
            predict_price({
                "city": "Tp Hồ Chí Minh",
                "ward": "Phường 1",
                "area": 0,
            })
        self.assertIn("Diện tích", str(ctx.exception))


class TestStreetValidation(unittest.TestCase):
    """3. Kiểm thử xác thực đường: có danh mục vs thiếu/rỗng dữ liệu."""

    def setUp(self):
        self.reliable_street_catalog = {
            "Tp Hồ Chí Minh": {
                "Phường Bến Nghé": ["Đường Đồng Khởi", "Đường Nguyễn Huệ", "Đường Lê Lợi"],
                "Phường 1": ["Đường Trần Hưng Đạo"],
            }
        }

    def test_street_validation_with_reliable_catalog_success(self):
        """Trường hợp có danh mục: đường đúng phường được xác nhận và đánh dấu verified."""
        validator = StreetValidator(street_catalog=self.reliable_street_catalog)
        self.assertTrue(validator.is_active)
        res = validator.validate("Tp Hồ Chí Minh", "Phường Bến Nghé", "Đường Đồng Khởi")
        self.assertTrue(res["verified"])
        self.assertEqual(res["status"], "verified")
        self.assertIn("đã được xác thực", res["message"])

    def test_street_validation_case_and_whitespace_insensitive(self):
        """So sánh đường không phân biệt hoa thường và khoảng trắng."""
        validator = StreetValidator(street_catalog=self.reliable_street_catalog)
        res = validator.validate("Tp Hồ Chí Minh", "Phường Bến Nghé", "  đường   đồng   khởi  ")
        self.assertTrue(res["verified"])
        self.assertEqual(res["status"], "verified")

    def test_street_validation_with_reliable_catalog_wrong_ward(self):
        """Trường hợp có danh mục: đường thuộc phường khác bị từ chối và báo rõ phường thật sự sở hữu."""
        validator = StreetValidator(street_catalog=self.reliable_street_catalog)
        with self.assertRaises(ValueError) as ctx:
            validator.validate("Tp Hồ Chí Minh", "Phường 1", "Đường Đồng Khởi")
        msg = str(ctx.exception)
        self.assertIn("Phường Bến Nghé", msg)
        self.assertIn("không thuộc phường/xã 'Phường 1'", msg)

    def test_street_validation_with_reliable_catalog_nonexistent(self):
        """Trường hợp có danh mục: đường không có trong danh mục của phường bị từ chối."""
        validator = StreetValidator(street_catalog=self.reliable_street_catalog)
        with self.assertRaises(ValueError) as ctx:
            validator.validate("Tp Hồ Chí Minh", "Phường 1", "Đường Không Có Tên")
        self.assertIn("không có trong danh mục đường của phường/xã 'Phường 1'", str(ctx.exception))

    def test_street_validation_empty_street_allowed(self):
        """Bỏ trống đường: cho phép tiếp tục như hiện tại."""
        validator = StreetValidator(street_catalog=self.reliable_street_catalog)
        res = validator.validate("Tp Hồ Chí Minh", "Phường 1", "")
        self.assertFalse(res["verified"])
        self.assertEqual(res["status"], "empty")

        res_none = validator.validate("Tp Hồ Chí Minh", "Phường 1", None)
        self.assertFalse(res_none["verified"])
        self.assertEqual(res_none["status"], "empty")

    def test_street_validation_with_empty_or_missing_catalog(self):
        """Nếu danh mục thiếu hoặc rỗng: không chặn tùy tiện và không tuyên bố đã xác thực."""
        validator_none = StreetValidator(street_catalog={})
        self.assertFalse(validator_none.is_active)
        res = validator_none.validate("Tp Hồ Chí Minh", "Phường 1", "Đường Tùy Ý")
        self.assertFalse(res["verified"])
        self.assertEqual(res["status"], "unverified")

    def test_load_street_catalog_file_missing_returns_empty(self):
        """File danh mục không tồn tại: trả về dict rỗng an toàn."""
        non_existent_path = Path("models/non_existent_street_catalog.json")
        catalog = load_street_catalog(non_existent_path)
        self.assertEqual(catalog, {})

    def test_load_street_catalog_malformed_json_raises_error(self):
        """File danh mục bị lỗi cú pháp JSON: ném lỗi rõ ràng, không âm thầm coi như hợp lệ."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            f.write("{ invalid json content : ")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_street_catalog(temp_path)
            self.assertIn("nội dung JSON không hợp lệ", str(ctx.exception))
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_street_catalog_invalid_structure_raises_error(self):
        """File danh mục sai cấu trúc (không phải dictionary): ném lỗi rõ ràng."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(["đây", "là", "list", "không", "phải", "dict"], f)
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_street_catalog(temp_path)
            self.assertIn("phải là một JSON object/dictionary", str(ctx.exception))
        finally:
            temp_path.unlink(missing_ok=True)

    def test_dataset_street_assessment_reveals_data_limit(self):
        """Đánh giá thực tế dataset.csv: phát hiện thiếu trường đường có cấu trúc (<1% thông tin)."""
        df_real = pd.read_csv("database/dataset.csv")
        assessment = get_street_data_assessment(df_real)
        self.assertFalse(assessment["is_sufficient_for_validation"])
        self.assertLess(assessment["coverage_percentage"], 1.0)
        self.assertIn("khuyết thiếu", assessment["limitation_reason"])
        self.assertIn("bổ sung", assessment["proposal"])


class TestUpdateSafety(unittest.TestCase):
    """4. Kiểm thử tính nhất quán và an toàn cập nhật (atomic update)."""

    def test_train_failure_preserves_old_model_and_catalog(self):
        """Lỗi trong quá trình train không làm mất mô hình và danh mục cũ."""
        initial_bundle = load_model()
        initial_catalog = get_active_location_catalog()
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(CATALOG_PATH.exists())

        invalid_bundle = {
            "model": "not_a_model",
            "slope": float("nan"),
            "intercept": 0.0,
            "feats": [],
            "algorithm": "broken",
            "mode": "sale",
            "accuracy": 0.0,
        }
        with self.assertRaises(Exception):
            save_model_bundle(invalid_bundle, MODEL_PATH)

        reloaded_bundle = reload_model()
        reloaded_catalog = get_active_location_catalog()
        self.assertIn("model", reloaded_bundle)
        self.assertEqual(list(reloaded_catalog.keys()), list(initial_catalog.keys()))
        self.assertEqual(reloaded_bundle["accuracy"], initial_bundle["accuracy"])


class TestFlaskEndpoints(unittest.TestCase):
    """5. Kiểm thử các endpoint Flask (dynamic catalog API & predict view)."""

    def setUp(self):
        self.client = app.test_client()
        clear_street_catalog_cache()

    def test_api_locations_endpoint(self):
        """API /api/locations trả về danh mục động và cờ xác thực đường."""
        resp = self.client.get("/api/locations")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("Tp Hồ Chí Minh", data["cities"])
        self.assertIn("Tp Hồ Chí Minh", data["locations"])
        self.assertGreater(len(data["locations"]["Tp Hồ Chí Minh"]), 50)
        self.assertIn("street_validation", data)

    def test_predict_view_post_valid_and_invalid_locations(self):
        """Kiểm thử form POST /predict: từ chối địa điểm sai và hiển thị lỗi trên HTML."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "testuser"

        # 1. GET /predict chứa danh mục động và không có mảng hard-code
        resp_get = self.client.get("/predict")
        self.assertEqual(resp_get.status_code, 200)
        html_get = resp_get.get_data(as_text=True)
        self.assertIn("LOCATIONS", html_get)
        self.assertNotIn("const WARDS =", html_get)

        # 2. POST /predict với thành phố không tồn tại
        resp_post_invalid = self.client.post("/predict", data={
            "city": "Thành phố Giả Định",
            "ward": "Phường 1",
            "area": "60",
            "bedrooms": "2",
            "bathrooms": "1",
        })
        self.assertEqual(resp_post_invalid.status_code, 200)
        html_post = resp_post_invalid.get_data(as_text=True)
        self.assertIn("không tồn tại trong danh mục", html_post)

        # 3. POST /predict với phường không thuộc thành phố
        resp_post_wrong_ward = self.client.post("/predict", data={
            "city": "Tp Hồ Chí Minh",
            "ward": "Phường Tràng Tiền",
            "area": "60",
            "bedrooms": "2",
            "bathrooms": "1",
        })
        self.assertEqual(resp_post_wrong_ward.status_code, 200)
        html_wrong_ward = resp_post_wrong_ward.get_data(as_text=True)
        self.assertIn("không tồn tại trong danh mục của thành phố", html_wrong_ward)

    def test_predict_view_with_street_catalog_lifecycle(self):
        """
        Kiểm thử tích hợp luồng Flask với file danh mục đường mẫu:
        - Đường hợp lệ -> thành công, hiển thị badge đã xác thực.
        - Đường thuộc phường khác -> từ chối, báo lỗi rõ ràng, giữ nguyên form.
        - Đường không có trong danh mục -> từ chối, hiển thị lỗi trong 'Không thể phân tích'.
        - Bỏ trống đường -> thành công.
        """
        sample_catalog_data = {
            "Tp Hồ Chí Minh": {
                "Phường Bến Nghé": ["Đường Đồng Khởi", "Đường Nguyễn Huệ"],
                "Phường 1": ["Đường Trần Hưng Đạo"]
            }
        }
        # Tạo file tạm models/street_catalog.json cho test
        sample_file = STREET_CATALOG_PATH
        existed_before = sample_file.exists()
        original_bytes = sample_file.read_bytes() if existed_before else None

        try:
            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(sample_catalog_data, f, ensure_ascii=False, indent=2)
            clear_street_catalog_cache()

            with self.client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["username"] = "testuser"

            # 1. Đường hợp lệ đúng phường: Phường Bến Nghé + Đường Đồng Khởi
            resp_valid = self.client.post("/predict", data={
                "city": "Tp Hồ Chí Minh",
                "ward": "Phường Bến Nghé",
                "street": "Đường Đồng Khởi",
                "area": "70",
                "bedrooms": "2",
                "bathrooms": "2",
            })
            self.assertEqual(resp_valid.status_code, 200)
            html_valid = resp_valid.get_data(as_text=True)
            self.assertIn("Phân tích thành công", html_valid)
            self.assertIn("Đã xác thực thuộc phường", html_valid)
            self.assertIn("Đường Đồng Khởi", html_valid)

            # 2. Đường thuộc phường khác: Phường 1 + Đường Đồng Khởi
            resp_wrong_ward = self.client.post("/predict", data={
                "city": "Tp Hồ Chí Minh",
                "ward": "Phường 1",
                "street": "Đường Đồng Khởi",
                "area": "70",
                "bedrooms": "2",
                "bathrooms": "2",
            })
            self.assertEqual(resp_wrong_ward.status_code, 200)
            html_wrong = html.unescape(resp_wrong_ward.get_data(as_text=True))
            self.assertIn("Không thể phân tích", html_wrong)
            self.assertIn("thuộc Phường Bến Nghé, không thuộc phường/xã 'Phường 1'", html_wrong)
            # Dữ liệu người dùng nhập phải được giữ nguyên trong input form
            self.assertIn('value="Đường Đồng Khởi"', html_wrong)
            self.assertIn('value="70"', html_wrong)

            # 3. Đường không có trong danh mục: Phường 1 + Đường Lê Lợi
            resp_not_in_cat = self.client.post("/predict", data={
                "city": "Tp Hồ Chí Minh",
                "ward": "Phường 1",
                "street": "Đường Lê Lợi",
                "area": "70",
                "bedrooms": "2",
                "bathrooms": "2",
            })
            self.assertEqual(resp_not_in_cat.status_code, 200)
            html_not_in_cat = html.unescape(resp_not_in_cat.get_data(as_text=True))
            self.assertIn("Không thể phân tích", html_not_in_cat)
            self.assertIn("không có trong danh mục đường của phường/xã 'Phường 1'", html_not_in_cat)
            self.assertIn('value="Đường Lê Lợi"', html_not_in_cat)

            # 4. Bỏ trống đường: cho phép dự đoán bình thường
            resp_empty_street = self.client.post("/predict", data={
                "city": "Tp Hồ Chí Minh",
                "ward": "Phường 1",
                "street": "",
                "area": "70",
                "bedrooms": "2",
                "bathrooms": "2",
            })
            self.assertEqual(resp_empty_street.status_code, 200)
            html_empty = resp_empty_street.get_data(as_text=True)
            self.assertIn("Phân tích thành công", html_empty)

        finally:
            # Khôi phục trạng thái ban đầu của file
            if existed_before and original_bytes is not None:
                sample_file.write_bytes(original_bytes)
            else:
                sample_file.unlink(missing_ok=True)
            clear_street_catalog_cache()

    def test_predict_view_when_street_catalog_missing_shows_unverified(self):
        """Khi file danh mục không tồn tại: dự đoán vẫn thành công và hiển thị 'Chưa đối chiếu danh mục đường'."""
        sample_file = STREET_CATALOG_PATH
        existed_before = sample_file.exists()
        original_bytes = sample_file.read_bytes() if existed_before else None
        if existed_before:
            sample_file.unlink(missing_ok=True)
        clear_street_catalog_cache()

        try:
            with self.client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["username"] = "testuser"

            resp = self.client.post("/predict", data={
                "city": "Tp Hồ Chí Minh",
                "ward": "Phường 1",
                "street": "Đường Nguyễn Trãi",
                "area": "60",
                "bedrooms": "2",
                "bathrooms": "1",
            })
            self.assertEqual(resp.status_code, 200)
            html = resp.get_data(as_text=True)
            self.assertIn("Phân tích thành công", html)
            self.assertIn("Chưa đối chiếu danh mục đường", html)
            self.assertIn("Đường Nguyễn Trãi", html)
        finally:
            if existed_before and original_bytes is not None:
                sample_file.write_bytes(original_bytes)
            clear_street_catalog_cache()


if __name__ == "__main__":
    unittest.main()
