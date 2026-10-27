from pathlib import Path
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from backend.auth import authenticate_user, authenticate_admin, register_user
from backend.location_service import get_active_location_catalog, StreetValidator

public_bp = Blueprint("public_views", __name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@public_bp.route("/")
def index():
    model_accuracy = "N/A"
    report_path = ROOT_DIR / "models" / "model_report.txt"
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Accuracy" in line:
                try:
                    model_accuracy = line.split(":")[-1].strip().replace("%", "")
                except Exception:
                    pass
    return render_template("index.html", model_accuracy=model_accuracy)


@public_bp.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("user_views.predict"))
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        confirm  = request.form["confirm_password"]
        fullname = request.form["fullname"].strip()
        email    = request.form["email"].strip()

        if len(password) < 6:
            flash("Mật khẩu phải có ít nhất 6 ký tự!", "error")
        elif password != confirm:
            flash("Mật khẩu không khớp!", "error")
        else:
            ok, msg = register_user(username, password, fullname, email)
            flash(msg, "success" if ok else "error")
            if ok:
                return redirect(url_for("public_views.login"))

    return render_template("register.html")


@public_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("user_views.predict"))
    if "admin" in session:
        return redirect(url_for("admin_views.admin_dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        is_admin = request.form.get("is_admin")

        if is_admin:
            admin = authenticate_admin(username, password)
            if admin:
                session["admin"] = admin["username"]
                return redirect(url_for("admin_views.admin_dashboard"))
            else:
                flash("Sai tên đăng nhập hoặc mật khẩu quản trị!", "error")
        else:
            user = authenticate_user(username, password)
            if user:
                session["user_id"]  = user["id"]
                session["username"] = user["username"]
                session["fullname"] = user.get("fullname", user["username"])
                return redirect(url_for("user_views.predict"))
            else:
                flash("Sai tên đăng nhập hoặc mật khẩu!", "error")

    return render_template("login.html")


@public_bp.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất thành công.", "success")
    return redirect(url_for("public_views.index"))


@public_bp.route("/api/locations")
def api_locations():
    catalog = get_active_location_catalog()
    validator = StreetValidator()
    return jsonify({
        "success": bool(catalog),
        "cities": list(catalog.keys()),
        "locations": catalog,
        "street_validation": {
            "active": validator.is_active,
            "message": (
                "Đã kích hoạt xác thực đường–phường." if validator.is_active else
                "Chưa bật xác thực đường–phường do tập dữ liệu chưa có danh mục đường có cấu trúc."
            )
        }
    })