import hashlib
from backend.connect_db import get_cursor


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str, fullname: str, email: str) -> tuple[bool, str]:
    try:
        with get_cursor() as (cursor, conn):
            cursor.execute("SELECT id FROM users WHERE username=%s OR email=%s", (username, email))
            if cursor.fetchone():
                return False, "Tên đăng nhập hoặc email đã tồn tại!"
            cursor.execute(
                "INSERT INTO users (username, password, fullname, email) VALUES (%s,%s,%s,%s)",
                (username, hash_password(password), fullname, email),
            )
        return True, "Đăng ký thành công!"
    except Exception as e:
        return False, str(e)


def authenticate_user(username: str, password: str):
    with get_cursor() as (cursor, _):
        cursor.execute(
            "SELECT id, username, fullname, email FROM users WHERE username=%s AND password=%s",
            (username, hash_password(password)),
        )
        return cursor.fetchone()


def authenticate_admin(username: str, password: str):
    """Admin password stored as plain (seeded). Hash on first real login if needed."""
    with get_cursor() as (cursor, _):
        cursor.execute(
            "SELECT id, username FROM admins WHERE username=%s AND password=%s",
            (username, password),
        )
        return cursor.fetchone()
