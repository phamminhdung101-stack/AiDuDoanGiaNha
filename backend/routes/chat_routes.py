"""
Blueprint định tuyến cho Chức năng Chatbot AI Tư vấn Bất động sản
HousePrice AI Platform
"""
from functools import wraps
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from backend.chatbot.service import RealEstateChatService
from backend.chatbot.db_service import ChatDatabaseService

chat_bp = Blueprint("chat_views", __name__)


def get_current_user_id():
    """Lấy ID người dùng hiện tại từ session (hỗ trợ cả user và admin)."""
    if "user_id" in session:
        return int(session["user_id"])
    if "admin" in session:
        return 999999
    return None


def chat_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = get_current_user_id()
        if not user_id:
            if request.path.startswith("/api/"):
                return jsonify({"success": False, "error": "Vui lòng đăng nhập để tiếp tục!", "code": "UNAUTHORIZED"}), 401
            flash("Vui lòng đăng nhập để sử dụng Chatbot AI tư vấn bất động sản!", "error")
            return redirect(url_for("public_views.login"))
        return f(*args, **kwargs)
    return decorated


@chat_bp.route("/chat")
@chat_login_required
def chat_page():
    """Trang giao diện chính của Chatbot AI tư vấn bất động sản."""
    user_id = get_current_user_id()
    sessions = ChatDatabaseService.get_user_sessions(user_id)
    active_session_id = request.args.get("session_id", type=int)

    # Nếu chưa có phiên nào, tự động tạo một phiên ban đầu
    if not sessions:
        new_id = ChatDatabaseService.create_chat_session(user_id, "Tư vấn bất động sản mới")
        sessions = ChatDatabaseService.get_user_sessions(user_id)
        active_session_id = new_id
    elif not active_session_id or not any(s["id"] == active_session_id for s in sessions):
        active_session_id = sessions[0]["id"]

    messages = ChatDatabaseService.get_session_messages(active_session_id, user_id)

    return render_template(
        "chat.html",
        sessions=sessions,
        active_session_id=active_session_id,
        messages=messages,
        user_id=user_id,
    )


@chat_bp.route("/api/chat/sessions", methods=["GET"])
@chat_login_required
def api_get_sessions():
    """Lấy danh sách các phiên trò chuyện của người dùng."""
    user_id = get_current_user_id()
    sessions = ChatDatabaseService.get_user_sessions(user_id)
    return jsonify({"success": True, "sessions": sessions})


@chat_bp.route("/api/chat/sessions", methods=["POST"])
@chat_login_required
def api_create_session():
    """Tạo một phiên trò chuyện mới."""
    user_id = get_current_user_id()
    data = request.get_json(silent=True) or {}
    session_name = data.get("name", "Cuộc trò chuyện mới").strip() or "Cuộc trò chuyện mới"
    new_id = ChatDatabaseService.create_chat_session(user_id, session_name)
    return jsonify({
        "success": True,
        "session_id": new_id,
        "session_name": session_name
    })


@chat_bp.route("/api/chat/sessions/<int:session_id>/messages", methods=["GET"])
@chat_login_required
def api_get_messages(session_id: int):
    """Lấy lịch sử tin nhắn của một phiên trò chuyện (kiểm tra quyền sở hữu)."""
    user_id = get_current_user_id()
    messages = ChatDatabaseService.get_session_messages(session_id, user_id)
    session_info = ChatDatabaseService.get_session(session_id, user_id)
    if not session_info:
        return jsonify({"success": False, "error": "Phiên trò chuyện không tồn tại hoặc bạn không có quyền truy cập."}), 403
    return jsonify({
        "success": True,
        "session": dict(session_info),
        "messages": messages
    })


@chat_bp.route("/api/chat/sessions/<int:session_id>/messages", methods=["POST"])
@chat_login_required
def api_send_message(session_id: int):
    """Gửi câu hỏi và nhận câu trả lời phân tích từ Chatbot AI."""
    user_id = get_current_user_id()
    session_info = ChatDatabaseService.get_session(session_id, user_id)
    if not session_info:
        return jsonify({"success": False, "error": "Phiên trò chuyện không hợp lệ hoặc bạn không có quyền truy cập."}), 403

    data = request.get_json(silent=True) or {}
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"success": False, "error": "Nội dung câu hỏi không được để trống."}), 400

    if len(user_text) > 2000:
        return jsonify({"success": False, "error": "Câu hỏi quá dài (tối đa 2,000 ký tự)."}), 400

    # 1. Lưu tin nhắn của người dùng
    user_msg_id = ChatDatabaseService.add_chat_message(session_id, "user", user_text)

    # 2. Cập nhật tên phiên nếu vẫn là tên mặc định
    if session_info.get("session_name") in ("Cuộc trò chuyện mới", "Tư vấn bất động sản mới"):
        clean_name = user_text[:35] + ("..." if len(user_text) > 35 else "")
        ChatDatabaseService.update_session_name(session_id, user_id, clean_name)

    # 3. Lấy lịch sử 6 tin nhắn gần nhất để làm ngữ cảnh trò chuyện
    history = ChatDatabaseService.get_session_messages(session_id, user_id, limit=6)

    # 4. Xử lý câu trả lời bằng RealEstateChatService
    ai_result = RealEstateChatService.process_message(user_text, history_messages=history)

    # 5. Lưu tin nhắn trả lời của AI
    ai_msg_id = ChatDatabaseService.add_chat_message(
        session_id=session_id,
        role="assistant",
        message=ai_result["message"],
        metadata=ai_result.get("metadata")
    )

    return jsonify({
        "success": True,
        "user_message_id": user_msg_id,
        "assistant_message_id": ai_msg_id,
        "role": "assistant",
        "message": ai_result["message"],
        "metadata": ai_result.get("metadata"),
        "provider": ai_result.get("provider", "ai"),
    })


@chat_bp.route("/api/chat/sessions/<int:session_id>", methods=["DELETE"])
@chat_login_required
def api_delete_session(session_id: int):
    """Xóa một phiên trò chuyện."""
    user_id = get_current_user_id()
    ok = ChatDatabaseService.delete_session(session_id, user_id)
    if ok:
        return jsonify({"success": True, "message": "Đã xóa phiên trò chuyện thành công."})
    return jsonify({"success": False, "error": "Không thể xóa phiên hoặc phiên không tồn tại."}), 400


@chat_bp.route("/api/chat/sessions/<int:session_id>", methods=["PUT"])
@chat_login_required
def api_update_session(session_id: int):
    """Đổi tên phiên trò chuyện."""
    user_id = get_current_user_id()
    data = request.get_json(silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"success": False, "error": "Tên phiên không hợp lệ."}), 400

    ok = ChatDatabaseService.update_session_name(session_id, user_id, new_name)
    if ok:
        return jsonify({"success": True, "message": "Đã đổi tên phiên thành công."})
    return jsonify({"success": False, "error": "Không thể đổi tên phiên."}), 400


@chat_bp.route("/api/chat/feedback", methods=["POST"])
@chat_login_required
def api_submit_feedback():
    """Gửi đánh giá và phản hồi chất lượng câu trả lời của AI (1 - 5 sao)."""
    user_id = get_current_user_id()
    data = request.get_json(silent=True) or {}
    msg_id = data.get("message_id")
    rating = data.get("rating")
    comment = data.get("comment", "")

    if not msg_id or not rating:
        return jsonify({"success": False, "error": "Thiếu message_id hoặc rating."}), 400

    try:
        fb_id = ChatDatabaseService.save_feedback(
            message_id=int(msg_id),
            user_id=user_id,
            rating=int(rating),
            comment=str(comment)
        )
        return jsonify({"success": True, "feedback_id": fb_id, "message": "Cảm ơn bạn đã đóng góp ý kiến đánh giá!"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi lưu đánh giá: {e}"}), 500


@chat_bp.route("/api/chat/suggestions", methods=["GET"])
def api_get_suggestions():
    """Danh sách các câu hỏi gợi ý nhanh cho người dùng."""
    return jsonify({
        "success": True,
        "suggestions": [
            {"label": "🎯 Định giá nhà", "prompt": "Nhà 70m2 tại Quận 12, TP.HCM giá 4 tỷ có hợp lý không?"},
            {"label": "🔍 Tìm nhà dưới 3 tỷ", "prompt": "Tìm nhà dưới 3 tỷ tại TP.HCM, diện tích trên 50m2 có 2 phòng ngủ"},
            {"label": "📊 Tính tỷ suất cho thuê", "prompt": "Nhà 4 tỷ cho thuê 15 triệu/tháng thì tỷ suất sinh lời và thời gian hoàn vốn thế nào?"},
            {"label": "🏦 Bảng tính vay trả góp", "prompt": "Nếu mua nhà 3.5 tỷ vay ngân hàng 70% trong 15 năm thì mỗi tháng trả bao nhiêu tiền?"},
            {"label": "🔬 Giải thích cơ sở định giá", "prompt": "Tại sao diện tích và vị trí đường phố lại ảnh hưởng lớn đến giá căn nhà?"},
        ]
    })
