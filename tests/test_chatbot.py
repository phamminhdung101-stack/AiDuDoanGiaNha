"""
Unit & Integration Tests for AI Real Estate Consulting Chatbot
HousePrice AI Platform
"""
import json
import unittest
from app import app
from backend.chatbot.service import RealEstateChatService
from backend.chatbot.db_service import ChatDatabaseService


class TestRealEstateChatbot(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_unauthorized_chat_page_redirects(self):
        """Người dùng chưa đăng nhập bị chuyển hướng về /login khi truy cập /chat."""
        resp = self.client.get("/chat")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers.get("Location", ""))

    def test_unauthorized_api_returns_401(self):
        """API yêu cầu xác thực trả về mã 401 nếu chưa đăng nhập."""
        resp = self.client.get("/api/chat/sessions")
        self.assertEqual(resp.status_code, 401)
        data = json.loads(resp.data)
        self.assertFalse(data.get("success"))
        self.assertEqual(data.get("code"), "UNAUTHORIZED")

    def test_authorized_chat_page_loads_200(self):
        """Người dùng đã đăng nhập truy cập /chat thành công 200 OK."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "testuser"

        resp = self.client.get("/chat")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Trợ lý AI Tư vấn Bất động sản", html)
        self.assertIn("Cuộc trò chuyện mới", html)
        self.assertIn("chatInput", html)

    def test_session_lifecycle_and_isolation(self):
        """Kiểm tra tạo, đổi tên, lấy tin nhắn, cách ly giữa người dùng và xóa session."""
        user_a = 101
        user_b = 102

        # 1. User A tạo session
        sess_a_id = ChatDatabaseService.create_chat_session(user_a, "Phiên của User A")
        self.assertIsNotNone(sess_a_id)

        # 2. User B không thể truy cập session của User A
        sess_info = ChatDatabaseService.get_session(sess_a_id, user_b)
        self.assertIsNone(sess_info)

        # 3. User A thêm tin nhắn
        msg_id = ChatDatabaseService.add_chat_message(sess_a_id, "user", "Nhà 80m2 tại Quận 1 bao nhiêu tiền?")
        self.assertIsNotNone(msg_id)

        # 4. User B không thể đọc tin nhắn của User A
        msgs_b = ChatDatabaseService.get_session_messages(sess_a_id, user_b)
        self.assertEqual(len(msgs_b), 0)

        # 5. User A đọc tin nhắn hợp lệ
        msgs_a = ChatDatabaseService.get_session_messages(sess_a_id, user_a)
        self.assertEqual(len(msgs_a), 1)
        self.assertEqual(msgs_a[0]["message"], "Nhà 80m2 tại Quận 1 bao nhiêu tiền?")

        # 6. Đổi tên phiên
        ok_rename = ChatDatabaseService.update_session_name(sess_a_id, user_a, "Định giá Quận 1")
        self.assertTrue(ok_rename)

        # 7. Xóa phiên
        ok_del = ChatDatabaseService.delete_session(sess_a_id, user_a)
        self.assertTrue(ok_del)
        self.assertIsNone(ChatDatabaseService.get_session(sess_a_id, user_a))

    def test_nlp_property_specs_extraction(self):
        """Kiểm tra trích xuất thông số kỹ thuật nhà đất từ câu hỏi tự nhiên."""
        query = "Căn nhà 75.5m2 tại Quận 12 giá 4.2 tỷ có 3 phòng ngủ và 2 wc có ổn không?"
        specs = RealEstateChatService.extract_property_specs(query)

        self.assertEqual(specs["area"], 75.5)
        self.assertEqual(specs["user_price"], 4_200_000_000)
        self.assertEqual(specs["bedrooms"], 3)
        self.assertEqual(specs["bathrooms"], 2)
        self.assertEqual(specs["district"], "Quận 12")
        self.assertEqual(specs["city"], "Tp Hồ Chí Minh")

    def test_price_prediction_consulting(self):
        """Kiểm tra tư vấn định giá nhà bằng Machine Learning kết hợp đánh giá mức giá người dùng."""
        query = "Nhà 70m2 tại Quận 12 giá 4 tỷ có hợp lý không?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("Kết quả Định giá Bất động sản AI", res["message"])
        self.assertIn("prediction", res["metadata"])
        pred = res["metadata"]["prediction"]
        self.assertGreater(pred["price"], 0)
        self.assertIn("Tỷ VNĐ", pred["price_formatted"])

    def test_house_search_intent(self):
        """Kiểm tra chức năng tìm kiếm bất động sản theo tiêu chí tự nhiên."""
        query = "Tìm nhà dưới 4 tỷ tại TP.HCM trên 40m2 có 2 phòng ngủ"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("houses", res["metadata"])
        # Nếu database/dataset có nhà phù hợp
        if res["metadata"]["houses"]:
            first_house = res["metadata"]["houses"][0]
            self.assertIn("title", first_house)
            self.assertIn("price_formatted", first_house)
            self.assertLessEqual(first_house["price"], 4_000_000_000)

    def test_investment_and_mortgage_analysis(self):
        """Kiểm tra phân tích tỷ suất cho thuê và bảng tính vay trả góp ngân hàng."""
        query = "Nhà 4 tỷ cho thuê 15 triệu/tháng thì tỷ suất thế nào, nếu vay 70% trong 15 năm trả góp bao nhiêu?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("investment", res["metadata"])
        inv = res["metadata"]["investment"]
        self.assertTrue(inv["success"])
        self.assertIn("rental", inv)
        self.assertIn("mortgage", inv)
        # Yield = 15tr * 12 / 4 tỷ = 4.5%
        self.assertAlmostEqual(inv["rental"]["yield_pct"], 4.5, places=1)
        self.assertGreater(inv["mortgage"]["monthly_payment"], 0)

    def test_explain_prediction_intent(self):
        """Kiểm tra giải thích cơ sở mô hình định giá AI."""
        query = "Giải thích cho tôi các yếu tố ảnh hưởng đến giá căn nhà?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("explanation", res["metadata"])
        exp = res["metadata"]["explanation"]
        self.assertIn("slope", exp)
        self.assertIn("area_effect", exp)
        self.assertIn("location_effect", exp)

    def test_feedback_api(self):
        """Kiểm tra API gửi đánh giá chất lượng câu trả lời (1 - 5 sao)."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "testuser"

        sess_id = ChatDatabaseService.create_chat_session(1, "Test Feedback")
        msg_id = ChatDatabaseService.add_chat_message(sess_id, "assistant", "Câu trả lời của AI")

        resp = self.client.post("/api/chat/feedback", json={
            "message_id": msg_id,
            "rating": 5,
            "comment": "Câu trả lời rất chi tiết và hữu ích!"
        })
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertIn("feedback_id", data)

    def test_suggestions_api(self):
        """Kiểm tra API danh sách câu hỏi gợi ý."""
        resp = self.client.get("/api/chat/suggestions")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data.get("success"))
        self.assertGreater(len(data.get("suggestions", [])), 0)

    # ── 10 TRƯỜNG HỢP KIỂM THỬ BẮT BUỘC THEO SECTION 9 ───────────────────────

    def test_case_1_vung_tau_not_hcm(self):
        """STT 1: 'Nhà ở Vũng Tàu 100m² giá bao nhiêu?' -> Không được trả về TP.HCM."""
        query = "Nhà ở Vũng Tàu 100m2 giá khoảng bao nhiêu?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("prediction", res["metadata"])
        pred = res["metadata"]["prediction"]

        # Phải nhận diện đúng Bà Rịa - Vũng Tàu, TUYỆT ĐỐI KHÔNG ĐƯỢC trả về TP.HCM
        self.assertEqual(pred["city"], "Bà Rịa - Vũng Tàu")
        self.assertNotEqual(pred["city"], "Tp Hồ Chí Minh")
        self.assertTrue(pred["is_city_level"])
        self.assertIn("Bà Rịa - Vũng Tàu", res["message"])
        self.assertNotIn("Tp Hồ Chí Minh", res["message"])

    def test_case_2_hcm_recognition(self):
        """STT 2: 'Nhà ở TP.HCM 100m² giá bao nhiêu?' -> Phải nhận diện đúng TP.HCM."""
        query = "Nhà ở TP.HCM 100m2 giá bao nhiêu?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("prediction", res["metadata"])
        pred = res["metadata"]["prediction"]
        self.assertEqual(pred["city"], "Tp Hồ Chí Minh")
        self.assertEqual(pred["area"], 100.0)

    def test_case_3_danang_recognition(self):
        """STT 3: 'Nhà ở Đà Nẵng 80m² giá bao nhiêu?' -> Không nhầm địa điểm."""
        query = "Nhà ở Đà Nẵng 80m2 giá bao nhiêu?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("prediction", res["metadata"])
        pred = res["metadata"]["prediction"]
        self.assertEqual(pred["city"], "Đà Nẵng")
        self.assertNotEqual(pred["city"], "Tp Hồ Chí Minh")
        self.assertEqual(pred["area"], 80.0)

    def test_case_4_hanoi_recognition(self):
        """STT 4: 'Nhà ở Hà Nội 90m² giá bao nhiêu?' -> Không nhầm địa điểm."""
        query = "Nhà ở Hà Nội 90m2 giá bao nhiêu?"
        res = RealEstateChatService.process_message(query)

        self.assertEqual(res["role"], "assistant")
        self.assertIn("prediction", res["metadata"])
        pred = res["metadata"]["prediction"]
        self.assertEqual(pred["city"], "Hà Nội")
        self.assertNotEqual(pred["city"], "Tp Hồ Chí Minh")
        self.assertEqual(pred["area"], 90.0)

    def test_case_5_district_12_hcm(self):
        """STT 5: 'Nhà ở Quận 12, TP.HCM 70m²' -> Nhận diện đúng quận và thành phố."""
        query = "Nhà ở Quận 12, TP.HCM 70m2"
        specs = RealEstateChatService.extract_property_specs(query)
        self.assertEqual(specs["city"], "Tp Hồ Chí Minh")
        self.assertEqual(specs["district"], "Quận 12")
        self.assertEqual(specs["area"], 70.0)

        res = RealEstateChatService.process_message(query)
        self.assertEqual(res["metadata"]["prediction"]["city"], "Tp Hồ Chí Minh")

    def test_case_6_vung_tau_specs_without_area(self):
        """STT 6: 'Nhà ở Vũng Tàu 2 tầng, 3 phòng ngủ' -> Trích xuất đúng thông tin."""
        query = "Tôi muốn định giá nhà ở Vũng Tàu 2 tầng, 3 phòng ngủ"
        specs = RealEstateChatService.extract_property_specs(query)

        self.assertEqual(specs["city"], "Bà Rịa - Vũng Tàu")
        self.assertEqual(specs["floors"], 2)
        self.assertEqual(specs["bedrooms"], 3)
        self.assertIsNone(specs["area"])

        res = RealEstateChatService.process_message(query)
        self.assertIn("missing_area", res["metadata"])
        self.assertIn("diện tích", res["message"].lower())

    def test_case_7_no_location_not_default_hcm(self):
        """STT 7: 'Nhà ở 100m² giá 4 tỷ có hợp lý không?' -> Không tự thêm địa điểm."""
        query = "Nhà ở 100m2 giá 4 tỷ có hợp lý không?"
        specs = RealEstateChatService.extract_property_specs(query)

        # Tuyệt đối không tự tiện gán TP.HCM khi người dùng không cung cấp
        self.assertIsNone(specs["city"])
        self.assertIsNone(specs["ward"])
        self.assertEqual(specs["area"], 100.0)
        self.assertEqual(specs["user_price"], 4_000_000_000)

        res = RealEstateChatService.process_message(query)
        self.assertIn("missing_location", res["metadata"])
        self.assertNotIn("Phường 1", res["message"])
        self.assertIn("vị trí", res["message"].lower())

    def test_case_8_vung_tau_then_danang(self):
        """STT 8: 'Nhà ở Vũng Tàu, sau đó hỏi nhà ở Đà Nẵng' -> Cập nhật đúng địa điểm mới."""
        history = [
            {"role": "user", "message": "Nhà ở Vũng Tàu 100m2 giá bao nhiêu?"},
            {
                "role": "assistant",
                "message": "Giá ước tính tại Bà Rịa - Vũng Tàu là 2.67 tỷ.",
                "metadata": {"specs": {"city": "Bà Rịa - Vũng Tàu", "area": 100.0}}
            }
        ]

        query2 = "Thế còn ở Đà Nẵng thì sao?"
        res2 = RealEstateChatService.process_message(query2, history_messages=history)

        self.assertEqual(res2["metadata"]["prediction"]["city"], "Đà Nẵng")
        self.assertNotEqual(res2["metadata"]["prediction"]["city"], "Bà Rịa - Vũng Tàu")
        self.assertEqual(res2["metadata"]["prediction"]["area"], 100.0)

    def test_case_9_unsupported_region(self):
        """STT 9: 'Nhà ở khu vực không có trong dữ liệu huấn luyện' -> Thông báo giới hạn hỗ trợ."""
        query = "Nhà ở Điện Biên 100m2 giá bao nhiêu?"
        specs = RealEstateChatService.extract_property_specs(query)
        self.assertEqual(specs["city"], "Điện Biên")

        res = RealEstateChatService.process_message(query)
        self.assertIn("unsupported_region", res["metadata"])
        self.assertEqual(res["metadata"]["unsupported_region"]["city"], "Điện Biên")
        # Không tự đổi thành TP.HCM
        self.assertNotIn("Tp Hồ Chí Minh", res["message"])
        self.assertIn("chưa đủ để định giá đáng tin cậy", res["message"])

    def test_case_10_missing_area(self):
        """STT 10: 'Người dùng không cung cấp diện tích' -> Hỏi bổ sung nếu cần thiết."""
        query = "Định giá giúp tôi nhà ở Phường Bến Nghé, Quận 1"
        specs = RealEstateChatService.extract_property_specs(query)
        self.assertIsNone(specs["area"])

        res = RealEstateChatService.process_message(query)
        self.assertIn("missing_area", res["metadata"])
        self.assertIn("diện tích", res["message"].lower())


if __name__ == "__main__":
    unittest.main()
