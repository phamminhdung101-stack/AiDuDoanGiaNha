"""
Unit and integration tests for Area Comparison and Property Map features
HousePrice AI Platform
"""
import json
import unittest
from app import app


class TestComparisonAndMapFeatures(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_compare_page_loads_200(self):
        """Trang so sánh giá nhà tải thành công 200 OK."""
        resp = self.client.get("/compare")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("So sánh Giá Bất động sản Đa Khu vực", html)
        self.assertIn("chartAvgPrice", html)

    def test_api_provinces(self):
        """API trả về danh sách các tỉnh/thành phố có dữ liệu."""
        resp = self.client.get("/api/areas/provinces")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertIn("Tp Hồ Chí Minh", data.get("provinces", []))

    def test_api_districts(self):
        """API trả về danh sách quận/huyện theo tỉnh/thành phố."""
        resp = self.client.get("/api/areas/districts?province=Tp%20H%E1%BB%93%20Ch%C3%AD%20Minh")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        districts = data.get("districts", [])
        self.assertIn("Quận 7", districts)
        self.assertIn("Quận Bình Thạnh", districts)

    def test_api_compare_areas_success(self):
        """API so sánh giá giữa nhiều khu vực trả về đầy đủ thẻ thống kê, biểu đồ và nhận xét."""
        resp = self.client.post("/api/areas/compare", json={
            "province": "Tp Hồ Chí Minh",
            "districts": ["Quận 7", "Quận Bình Thạnh", "Thành phố Thủ Đức", "Quận Bình Tân"]
        })
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertEqual(len(data.get("areas", [])), 4)

        # Kiểm tra cấu trúc từng khu vực
        for a in data["areas"]:
            self.assertIn("name", a)
            self.assertIn("avg_price", a)
            self.assertIn("avg_price_per_m2", a)
            self.assertIn("price_formatted", a)
            self.assertIn("ppm_formatted", a)
            self.assertIn("color", a)
            self.assertGreater(a["count"], 0)

        # Kiểm tra biểu đồ
        charts = data.get("charts", {})
        self.assertIn("avg_price", charts)
        self.assertIn("price_per_m2", charts)
        self.assertIn("price_range", charts)
        self.assertIn("samples", charts)

        # Kiểm tra nhận xét tự động
        insights = data.get("insights", [])
        self.assertGreater(len(insights), 0)

    def test_api_compare_areas_empty_districts_returns_400(self):
        """API so sánh báo lỗi 400 nếu danh sách khu vực rỗng."""
        resp = self.client.post("/api/areas/compare", json={
            "province": "Tp Hồ Chí Minh",
            "districts": []
        })
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertFalse(data.get("success"))

    def test_map_page_loads_200(self):
        """Trang bản đồ giá bất động sản tải thành công 200 OK."""
        resp = self.client.get("/map")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Bản đồ Phân bố Giá Bất động sản Toàn quốc", html)
        self.assertIn("propertyMap", html)

    def test_api_properties_map_data(self):
        """API bản đồ trả về danh sách Markers bất động sản, Heatmap và Price Tiers."""
        resp = self.client.get("/api/properties/map?province=Tp%20H%E1%BB%93%20Ch%C3%AD%20Minh&limit=50")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertGreater(len(data.get("items", [])), 0)
        self.assertGreater(len(data.get("heatmap", [])), 0)
        self.assertEqual(len(data.get("tiers", [])), 5)

        # Kiểm tra tính toàn vẹn của marker
        sample = data["items"][0]
        self.assertIn("id", sample)
        self.assertIn("price", sample)
        self.assertIn("price_formatted", sample)
        self.assertIn("price_per_m2", sample)
        self.assertIn("lat", sample)
        self.assertIn("lng", sample)
        self.assertIn("color", sample)
        self.assertIn("tier_id", sample)

        # Kiểm tra tọa độ trung tâm mặc định TP.HCM theo yêu cầu người dùng
        center = data.get("center", {})
        self.assertAlmostEqual(center.get("lat"), 10.8231, places=3)
        self.assertAlmostEqual(center.get("lng"), 106.6297, places=3)
        self.assertEqual(center.get("zoom"), 11)

    def test_api_map_config(self):
        """API cấu hình bản đồ Leaflet OpenStreetMap."""
        resp = self.client.get("/api/map/config")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("provider"), "leaflet_osm")
        self.assertIn("default_center", data)
        self.assertEqual(data["default_center"]["zoom"], 11)

    def test_map_page_uses_leaflet_and_no_api_key(self):
        """Trang map sử dụng Leaflet & OpenStreetMap, không còn ô nhập Google Maps API Key."""
        resp = self.client.get("/map")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        # Kiểm tra Leaflet & OpenStreetMap
        self.assertIn("leaflet.js", html)
        self.assertIn("leaflet.css", html)
        self.assertIn("leaflet.markercluster", html)
        self.assertIn("leaflet-heat", html)
        self.assertIn("tile.openstreetmap.org", html)
        self.assertNotIn("carto", html.lower())
        self.assertIn('id="propertyMap"', html)
        self.assertIn("DEFAULT_CENTER", html)
        # Đảm bảo đã xóa hoàn toàn giao diện và logic nhập Google Maps API Key
        self.assertNotIn('id="apiKeyModal"', html)
        self.assertNotIn('id="directApiKeyInput"', html)
        self.assertNotIn("GOOGLE_MAPS_KEY", html)
        self.assertNotIn("loadGoogleMapsScript", html)

    def test_api_geo_stats(self):
        """API thống kê địa lý theo khu vực hoạt động chính xác."""
        resp = self.client.get("/api/properties/geo-stats?province=Tp%20H%E1%BB%93%20Ch%C3%AD%20Minh&district=Qu%E1%BA%ADn%207")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertGreater(data.get("total_properties", 0), 0)
        self.assertGreater(data.get("avg_price", 0), 0)


if __name__ == "__main__":
    unittest.main()
