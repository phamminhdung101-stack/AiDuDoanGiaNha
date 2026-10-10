from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from backend.predict import reload_model
from backend.data_service import (
    add_house, delete_house, get_admin_stats,
    get_all_houses, get_house, get_recent_predictions, update_house
)

admin_bp = Blueprint("admin_views", __name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "admin" not in session:
            flash("Yêu cầu quyền Quản trị viên!", "error")
            return redirect(url_for("public_views.login"))
        return f(*args, **kwargs)
    return decorated


@admin_bp.route("/admin/dashboard")
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


@admin_bp.route("/admin/houses")
@admin_required
def admin_houses():
    search   = request.args.get("search", "").strip()
    page     = int(request.args.get("page", 1))
    per_page = 15
    houses, total = get_all_houses(search=search, page=page, per_page=per_page)
    return render_template("admin/houses.html",
                           houses=houses, total=total,
                           page=page, per_page=per_page, search=search)


@admin_bp.route("/admin/houses/add", methods=["GET", "POST"])
@admin_required
def admin_houses_add():
    if request.method == "POST":
        add_house(request.form.to_dict())
        flash("Đã thêm dữ liệu nhà thành công!", "success")
        return redirect(url_for("admin_views.admin_houses"))
    return render_template("admin/house_form.html", house=None, action="add")


@admin_bp.route("/admin/houses/edit/<int:house_id>", methods=["GET", "POST"])
@admin_required
def admin_houses_edit(house_id):
    house = get_house(house_id)
    if not house:
        flash("Không tìm thấy dữ liệu nhà!", "error")
        return redirect(url_for("admin_views.admin_houses"))
    if request.method == "POST":
        update_house(house_id, request.form.to_dict())
        flash("Đã cập nhật dữ liệu nhà!", "success")
        return redirect(url_for("admin_views.admin_houses"))
    return render_template("admin/house_form.html", house=house, action="edit")


@admin_bp.route("/admin/houses/delete/<int:house_id>")
@admin_required
def admin_houses_delete(house_id):
    delete_house(house_id)
    flash("Đã xóa dữ liệu nhà!", "success")
    return redirect(url_for("admin_views.admin_houses"))


@admin_bp.route("/admin/train", methods=["GET", "POST"])
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
    except Exception:
        train_data = {"total": 0, "valid": 0, "invalid": 0, "last_updated": "N/A"}

    return render_template("admin/train.html",
                           train_results=train_results,
                           model_report=model_report,
                           train_data=train_data)


@admin_bp.route("/admin/model")
@admin_required
def admin_model():
    return redirect(url_for("admin_views.admin_train"))