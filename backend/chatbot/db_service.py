"""
Dịch vụ Quản lý Cơ sở dữ liệu cho Chatbot AI (Chatbot Database Service)
HousePrice AI Platform
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from backend.connect_db import get_cursor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SQLITE_FALLBACK_PATH = ROOT_DIR / "models" / "chat_cache.db"


class ChatDatabaseService:
    @staticmethod
    @contextmanager
    def _sqlite_conn():
        SQLITE_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(SQLITE_FALLBACK_PATH))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id INTEGER NOT NULL,
                        user_id INTEGER NOT NULL,
                        rating INTEGER NOT NULL,
                        comment TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            yield conn
        finally:
            conn.close()

    @classmethod
    def create_chat_session(cls, user_id: int, session_name: str = "Cuộc trò chuyện mới") -> int:
        """Tạo một phiên trò chuyện mới cho người dùng."""
        try:
            with get_cursor() as (cursor, _):
                cursor.execute(
                    "INSERT INTO chat_sessions (user_id, session_name) VALUES (%s, %s)",
                    (user_id, session_name)
                )
                return cursor.lastrowid
        except Exception:
            # Fallback sang SQLite lưu trữ an toàn khi MySQL offline
            with cls._sqlite_conn() as conn:
                with conn:
                    cur = conn.execute(
                        "INSERT INTO chat_sessions (user_id, session_name, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                        (user_id, session_name)
                    )
                    return cur.lastrowid

    @classmethod
    def get_user_sessions(cls, user_id: int) -> list[dict]:
        """Lấy danh sách tất cả các phiên trò chuyện của người dùng."""
        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    SELECT id, user_id, session_name, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                """, (user_id,))
                rows = cursor.fetchall()
            for r in rows:
                if r.get("created_at") and hasattr(r["created_at"], "strftime"):
                    r["created_at"] = r["created_at"].strftime("%d/%m/%Y %H:%M")
                if r.get("updated_at") and hasattr(r["updated_at"], "strftime"):
                    r["updated_at"] = r["updated_at"].strftime("%d/%m/%Y %H:%M")
            return rows
        except Exception:
            with cls._sqlite_conn() as conn:
                cur = conn.execute("""
                    SELECT id, user_id, session_name, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                """, (user_id,))
                return [dict(row) for row in cur.fetchall()]

    @classmethod
    def get_session(cls, session_id: int, user_id: int) -> dict | None:
        """Kiểm tra quyền sở hữu và lấy thông tin phiên trò chuyện."""
        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    SELECT id, user_id, session_name, created_at, updated_at
                    FROM chat_sessions
                    WHERE id = %s AND user_id = %s
                """, (session_id, user_id))
                return cursor.fetchone()
        except Exception:
            with cls._sqlite_conn() as conn:
                cur = conn.execute("""
                    SELECT id, user_id, session_name, created_at, updated_at
                    FROM chat_sessions
                    WHERE id = ? AND user_id = ?
                """, (session_id, user_id))
                row = cur.fetchone()
                return dict(row) if row else None

    @classmethod
    def update_session_name(cls, session_id: int, user_id: int, new_name: str) -> bool:
        """Đổi tên phiên trò chuyện."""
        clean_name = str(new_name).strip()[:200]
        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    UPDATE chat_sessions
                    SET session_name = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND user_id = %s
                """, (clean_name, session_id, user_id))
                return cursor.rowcount > 0
        except Exception:
            with cls._sqlite_conn() as conn:
                with conn:
                    cur = conn.execute("""
                        UPDATE chat_sessions
                        SET session_name = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND user_id = ?
                    """, (clean_name, session_id, user_id))
                    return cur.rowcount > 0

    @classmethod
    def delete_session(cls, session_id: int, user_id: int) -> bool:
        """Xóa phiên trò chuyện và toàn bộ tin nhắn liên quan (Cascade)."""
        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    DELETE FROM chat_feedback WHERE message_id IN (
                        SELECT id FROM chat_messages WHERE session_id = %s
                    )
                """, (session_id,))
                cursor.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
                cursor.execute("DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
                return cursor.rowcount > 0
        except Exception:
            with cls._sqlite_conn() as conn:
                with conn:
                    conn.execute("""
                        DELETE FROM chat_feedback WHERE message_id IN (
                            SELECT id FROM chat_messages WHERE session_id = ?
                        )
                    """, (session_id,))
                    conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                    cur = conn.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
                    return cur.rowcount > 0

    @classmethod
    def add_chat_message(cls, session_id: int, role: str, message: str, metadata: dict | None = None) -> int:
        """Thêm tin nhắn vào phiên và cập nhật thời gian hoạt động của phiên."""
        meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else None
        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    INSERT INTO chat_messages (session_id, role, message, metadata)
                    VALUES (%s, %s, %s, %s)
                """, (session_id, role, message, meta_str))
                msg_id = cursor.lastrowid
                cursor.execute(
                    "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (session_id,)
                )
                return msg_id
        except Exception:
            with cls._sqlite_conn() as conn:
                with conn:
                    cur = conn.execute("""
                        INSERT INTO chat_messages (session_id, role, message, metadata)
                        VALUES (?, ?, ?, ?)
                    """, (session_id, role, message, meta_str))
                    msg_id = cur.lastrowid
                    conn.execute(
                        "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (session_id,)
                    )
                    return msg_id

    @classmethod
    def get_session_messages(cls, session_id: int, user_id: int, limit: int = 50) -> list[dict]:
        """Lấy danh sách tin nhắn của phiên (kiểm tra quyền sở hữu session)."""
        session = cls.get_session(session_id, user_id)
        if not session:
            return []

        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    SELECT id, session_id, role, message, metadata, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY id ASC
                    LIMIT %s
                """, (session_id, limit))
                rows = cursor.fetchall()
            for r in rows:
                if r.get("created_at") and hasattr(r["created_at"], "strftime"):
                    r["created_at"] = r["created_at"].strftime("%H:%M %d/%m")
                if r.get("metadata") and isinstance(r["metadata"], str):
                    try:
                        r["metadata"] = json.loads(r["metadata"])
                    except Exception:
                        r["metadata"] = None
            return rows
        except Exception:
            with cls._sqlite_conn() as conn:
                cur = conn.execute("""
                    SELECT id, session_id, role, message, metadata, created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                """, (session_id, limit))
                res = []
                for row in cur.fetchall():
                    d = dict(row)
                    if d.get("metadata") and isinstance(d["metadata"], str):
                        try:
                            d["metadata"] = json.loads(d["metadata"])
                        except Exception:
                            d["metadata"] = None
                    res.append(d)
                return res

    @classmethod
    def save_feedback(cls, message_id: int, user_id: int, rating: int, comment: str = "") -> int:
        """Lưu đánh giá câu trả lời (1 đến 5 sao) và nhận xét của người dùng."""
        rating = max(1, min(5, int(rating)))
        comment = str(comment or "").strip()
        try:
            with get_cursor() as (cursor, _):
                cursor.execute("""
                    INSERT INTO chat_feedback (message_id, user_id, rating, comment)
                    VALUES (%s, %s, %s, %s)
                """, (message_id, user_id, rating, comment))
                return cursor.lastrowid
        except Exception:
            with cls._sqlite_conn() as conn:
                with conn:
                    cur = conn.execute("""
                        INSERT INTO chat_feedback (message_id, user_id, rating, comment)
                        VALUES (?, ?, ?, ?)
                    """, (message_id, user_id, rating, comment))
                    return cur.lastrowid
