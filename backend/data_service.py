from backend.connect_db import get_cursor


# ── Prediction History ─────────────────────────────────────────────────

def save_prediction(user_id: int, form_data: dict, result: dict):
    with get_cursor() as (cursor, _):
        cursor.execute("""
            INSERT INTO predictions
                (user_id, city, district, street, area, house_type,
                 bedrooms, bathrooms, floors, pred_price, min_price, max_price, confidence)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            user_id,
            form_data.get("city", ""),
            form_data.get("district", ""),
            form_data.get("street", ""),
            float(form_data.get("area", 0) or 0),
            form_data.get("house_type", ""),
            int(form_data.get("bedrooms", 0) or 0),
            int(form_data.get("bathrooms", 0) or 0),
            int(form_data.get("floors", 0) or 0),
            result["price"],
            result["min_price"],
            result["max_price"],
            result["confidence"],
        ))


def get_user_history(user_id: int) -> list:
    with get_cursor() as (cursor, _):
        cursor.execute("""
            SELECT id, city, district, street, area, house_type,
                   bedrooms, bathrooms, floors, pred_price, min_price, max_price,
                   confidence, created_at
            FROM predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].strftime("%d/%m/%Y %H:%M")
        r["house_type_label"] = r.get("house_type", "").replace("_", " ").title()
    return rows


def delete_prediction(prediction_id: int, user_id: int):
    with get_cursor() as (cursor, _):
        cursor.execute(
            "DELETE FROM predictions WHERE id=%s AND user_id=%s",
            (prediction_id, user_id),
        )


# ── Houses (Admin) ─────────────────────────────────────────────────────

def get_all_houses(search: str = "", page: int = 1, per_page: int = 15):
    offset = (page - 1) * per_page
    like   = f"%{search}%"
    with get_cursor() as (cursor, _):
        if search:
            cursor.execute("""
                SELECT COUNT(*) as n FROM houses
                WHERE title LIKE %s OR city LIKE %s OR district LIKE %s
                   OR street LIKE %s OR house_type LIKE %s
            """, (like,)*5)
        else:
            cursor.execute("SELECT COUNT(*) as n FROM houses")
        total = cursor.fetchone()["n"]

        if search:
            cursor.execute("""
                SELECT * FROM houses
                WHERE title LIKE %s OR city LIKE %s OR district LIKE %s
                   OR street LIKE %s OR house_type LIKE %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
            """, (like,)*5 + (per_page, offset))
        else:
            cursor.execute(
                "SELECT * FROM houses ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (per_page, offset)
            )
        rows = cursor.fetchall()

    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].strftime("%d/%m/%Y")
    return rows, total


def get_house(house_id: int) -> dict | None:
    with get_cursor() as (cursor, _):
        cursor.execute("SELECT * FROM houses WHERE id=%s", (house_id,))
        return cursor.fetchone()


def add_house(form_data: dict) -> int:
    with get_cursor() as (cursor, _):
        cursor.execute("""
            INSERT INTO houses
                (title, city, district, street, area, floors, bedrooms, bathrooms,
                 frontage, road_width, house_type, direction, legal, price)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            form_data.get("title", ""),
            form_data.get("city", ""),
            form_data.get("district", ""),
            form_data.get("street", ""),
            float(form_data.get("area", 0) or 0),
            int(form_data.get("floors", 0) or 0),
            int(form_data.get("bedrooms", 0) or 0),
            int(form_data.get("bathrooms", 0) or 0),
            float(form_data.get("frontage", 0) or 0),
            float(form_data.get("road_width", 0) or 0),
            form_data.get("house_type", ""),
            form_data.get("direction", ""),
            form_data.get("legal", ""),
            float(form_data.get("price", 0) or 0),
        ))
        return cursor.lastrowid


def update_house(house_id: int, form_data: dict):
    with get_cursor() as (cursor, _):
        cursor.execute("""
            UPDATE houses SET
                title=%s, city=%s, district=%s, street=%s,
                area=%s, floors=%s, bedrooms=%s, bathrooms=%s,
                frontage=%s, road_width=%s, house_type=%s,
                direction=%s, legal=%s, price=%s
            WHERE id=%s
        """, (
            form_data.get("title", ""),
            form_data.get("city", ""),
            form_data.get("district", ""),
            form_data.get("street", ""),
            float(form_data.get("area", 0) or 0),
            int(form_data.get("floors", 0) or 0),
            int(form_data.get("bedrooms", 0) or 0),
            int(form_data.get("bathrooms", 0) or 0),
            float(form_data.get("frontage", 0) or 0),
            float(form_data.get("road_width", 0) or 0),
            form_data.get("house_type", ""),
            form_data.get("direction", ""),
            form_data.get("legal", ""),
            float(form_data.get("price", 0) or 0),
            house_id,
        ))


def delete_house(house_id: int):
    with get_cursor() as (cursor, _):
        cursor.execute("DELETE FROM houses WHERE id=%s", (house_id,))


# ── Admin Stats ────────────────────────────────────────────────────────

def get_admin_stats() -> dict:
    report_path = __import__('pathlib').Path(__file__).resolve().parent.parent / "models" / "model_report.txt"
    accuracy = "N/A"
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Accuracy" in line:
                try:
                    accuracy = line.split(":")[-1].strip()
                except:
                    pass

    with get_cursor() as (cursor, _):
        cursor.execute("SELECT COUNT(*) as n FROM houses")
        total_houses = cursor.fetchone()["n"]
        cursor.execute("SELECT COUNT(*) as n FROM predictions")
        total_preds  = cursor.fetchone()["n"]
        cursor.execute("SELECT COUNT(*) as n FROM users")
        total_users  = cursor.fetchone()["n"]

    return {
        "total_houses": total_houses,
        "predictions":  total_preds,
        "users":        total_users,
        "accuracy":     accuracy,
    }


def get_recent_predictions(limit: int = 8) -> list:
    with get_cursor() as (cursor, _):
        cursor.execute("""
            SELECT p.id, u.username, p.city, p.district, p.area, p.pred_price, p.created_at
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].strftime("%d/%m/%Y %H:%M")
    return rows
