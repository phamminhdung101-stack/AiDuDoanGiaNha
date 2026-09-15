import mysql.connector
from contextlib import contextmanager

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "house_prediction",
    "charset": "utf8mb4",
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


@contextmanager
def get_cursor(dictionary=True):
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=dictionary)
        yield cursor, conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def init_db():
    """Create database and tables if they don't exist."""
    # Connect without specifying DB first
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        charset=DB_CONFIG["charset"],
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.execute(f"USE {DB_CONFIG['database']}")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            username    VARCHAR(64) UNIQUE NOT NULL,
            password    VARCHAR(256) NOT NULL,
            fullname    VARCHAR(128) NOT NULL,
            email       VARCHAR(128) UNIQUE NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            username    VARCHAR(64) UNIQUE NOT NULL,
            password    VARCHAR(256) NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS houses (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            title       VARCHAR(256),
            city        VARCHAR(128),
            district    VARCHAR(128),
            street      VARCHAR(256),
            area        FLOAT,
            floors      INT,
            bedrooms    INT,
            bathrooms   INT,
            frontage    FLOAT,
            road_width  FLOAT,
            house_type  VARCHAR(64),
            direction   VARCHAR(64),
            legal       VARCHAR(64),
            price       FLOAT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            user_id     INT NOT NULL,
            city        VARCHAR(128),
            district    VARCHAR(128),
            street      VARCHAR(256),
            area        FLOAT,
            house_type  VARCHAR(64),
            bedrooms    INT,
            bathrooms   INT,
            floors      INT,
            pred_price  FLOAT,
            min_price   FLOAT,
            max_price   FLOAT,
            confidence  FLOAT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Seed default admin if not exists
    cursor.execute("SELECT id FROM admins WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO admins (username, password) VALUES ('admin', 'admin123')")

    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()
