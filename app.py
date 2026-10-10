import os
import sys
from pathlib import Path
from flask import Flask

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

app.secret_key = os.environ.get("SECRET_KEY", "houseai-secret-2024-change-in-prod")

from backend.connect_db import init_db
try:
    init_db()
except Exception as e:
    print(f"[WARN] DB init failed: {e}")

from backend.routes import register_routes
register_routes(app)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)