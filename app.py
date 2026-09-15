import os
import sys
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

app.secret_key = os.environ.get("SECRET_KEY", "houseai-secret-2024-change-in-prod")

from backend.connect_db import init_db
from backend.auth import authenticate_admin, authenticate_user, register_user
from backend.predict import predict_price, reload_model
from backend.data_service import (
    add_house, delete_house, get_admin_stats,
    get_all_houses, get_house, get_recent_predictions,
    get_user_history, save_prediction, update_house, delete_prediction,
)

try:
    init_db()
except Exception as e:
    print(f"[WARN] DB init failed: {e}")

# =========================
# DECORATORS
# =========================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Vui lòng đăng nhập để tiếp tục!", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin" not in session:
            flash("Yêu cầu quyền Quản trị viên!", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# =========================
# PUBLIC ROUTES
# =========================
@app.route("/")
def index():
    model_accuracy = "N/A"
    report_path = ROOT_DIR / "models" / "model_report.txt"
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Accuracy" in line:
                try:
                    model_accuracy = line.split(":")[-1].strip().replace("%", "")
                except:
                    pass
    return render_template("index.html", model_accuracy=model_accuracy)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("predict"))
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
                return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("predict"))
    if "admin" in session:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        is_admin = request.form.get("is_admin")

        if is_admin:
            admin = authenticate_admin(username, password)
            if admin:
                session["admin"] = admin["username"]
                return redirect(url_for("admin_dashboard"))
            else:
                flash("Sai tên đăng nhập hoặc mật khẩu quản trị!", "error")
        else:
            user = authenticate_user(username, password)
            if user:
                session["user_id"]  = user["id"]
                session["username"] = user["username"]
                session["fullname"] = user.get("fullname", user["username"])
                return redirect(url_for("predict"))
            else:
                flash("Sai tên đăng nhập hoặc mật khẩu!", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất thành công.", "success")
    return redirect(url_for("index"))

# =========================
# USER ROUTES
# =========================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result     = None
    error_msg  = None
    form_data  = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            result = predict_price(form_data)
            result["timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            save_prediction(session["user_id"], form_data, result)
        except FileNotFoundError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"Lỗi dự đoán: {e}"

    return render_template("predict.html", result=result, form_data=form_data, error_msg=error_msg)


@app.route("/history")
@login_required
def history():
    data = get_user_history(session["user_id"])
    return render_template("history.html", history=data)


@app.route("/history/delete/<int:pred_id>")
@login_required
def history_delete(pred_id):
    delete_prediction(pred_id, session["user_id"])
    flash("Đã xóa lịch sử dự đoán.", "success")
    return redirect(url_for("history"))

# =========================
# ADMIN ROUTES
# =========================
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    stats        = get_admin_stats()
    recent_preds = get_recent_predictions(limit=8)
    model_report = ""
    report_path  = ROOT_DIR / "models" / "model_report.txt"
    if report_path.exists():
        model_report = report_path.read_text(encoding="utf-8")
    return render_template("admin/dashboard.html", stats=stats,
                           recent_preds=recent_preds, model_report=model_report)


@app.route("/admin/houses")
@admin_required
def admin_houses():
    search   = request.args.get("search", "").strip()
    page     = int(request.args.get("page", 1))
    per_page = 15
    houses, total = get_all_houses(search=search, page=page, per_page=per_page)
    return render_template("admin/houses.html",
                           houses=houses, total=total,
                           page=page, per_page=per_page, search=search)


@app.route("/admin/houses/add", methods=["GET", "POST"])
@admin_required
def admin_houses_add():
    if request.method == "POST":
        add_house(request.form.to_dict())
        flash("Đã thêm dữ liệu nhà thành công!", "success")
        return redirect(url_for("admin_houses"))
    return render_template("admin/house_form.html", house=None, action="add")


@app.route("/admin/houses/edit/<int:house_id>", methods=["GET", "POST"])
@admin_required
def admin_houses_edit(house_id):
    house = get_house(house_id)
    if not house:
        flash("Không tìm thấy dữ liệu nhà!", "error")
        return redirect(url_for("admin_houses"))
    if request.method == "POST":
        update_house(house_id, request.form.to_dict())
        flash("Đã cập nhật dữ liệu nhà!", "success")
        return redirect(url_for("admin_houses"))
    return render_template("admin/house_form.html", house=house, action="edit")


@app.route("/admin/houses/delete/<int:house_id>")
@admin_required
def admin_houses_delete(house_id):
    delete_house(house_id)
    flash("Đã xóa dữ liệu nhà!", "success")
    return redirect(url_for("admin_houses"))


@app.route("/admin/train", methods=["GET", "POST"])
@admin_required
def admin_train():
    train_results = None
    model_report  = ""
    report_path   = ROOT_DIR / "models" / "model_report.txt"

    if request.method == "POST":
        from backend.train_model import train as run_train
        try:
            train_results = run_train(
                algorithm   = request.form.get("algorithm", "gradient_boosting"),
                test_size   = float(request.form.get("test_size", 20)) / 100,
                n_estimators= int(request.form.get("n_estimators", 200)),
                max_depth   = int(request.form.get("max_depth", 5)),
                mode        = request.form.get("mode", "sale"),
            )
            reload_model()
            flash("Huấn luyện mô hình thành công!", "success")
        except Exception as e:
            flash(f"Lỗi huấn luyện: {e}", "error")

    if report_path.exists():
        model_report = report_path.read_text(encoding="utf-8")

    from backend.train_model import load_dataset, clean_and_split
    try:
        df_raw   = load_dataset()
        df_clean = clean_and_split(df_raw, mode="sale")
        train_data = {
            "total":        len(df_raw),
            "valid":        len(df_clean),
            "invalid":      len(df_raw) - len(df_clean),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
        }
    except:
        train_data = {"total": 0, "valid": 0, "invalid": 0, "last_updated": "N/A"}

    return render_template("admin/train.html",
                           train_results=train_results,
                           model_report=model_report,
                           train_data=train_data)


# /admin/model → alias đến train (cập nhật mô hình = train lại)
@app.route("/admin/model")
@admin_required
def admin_model():
    return redirect(url_for("admin_train"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
