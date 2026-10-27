from datetime import datetime
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from backend.predict import predict_price
from backend.data_service import save_prediction, get_user_history, delete_prediction
from backend.location_service import get_active_location_catalog

user_bp = Blueprint("user_views", __name__)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Vui lòng đăng nhập để tiếp tục!", "error")
            return redirect(url_for("public_views.login"))
        return f(*args, **kwargs)
    return decorated


@user_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result    = None
    error_msg = None
    form_data = {}

    catalog = get_active_location_catalog()
    cities = list(catalog.keys())

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            result = predict_price(form_data)
            result["timestamp"] = datetime.now().strftime("%d/%m/%Y %H:%M")
            try:
                save_prediction(session["user_id"], form_data, result)
            except Exception as db_err:
                print(f"[WARN] Không thể lưu lịch sử dự đoán: {db_err}")
        except ValueError as e:
            error_msg = str(e)
        except FileNotFoundError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"Lỗi dự đoán: {e}"

    return render_template(
        "predict.html",
        result=result,
        form_data=form_data,
        error_msg=error_msg,
        locations=catalog,
        cities=cities,
    )


@user_bp.route("/history")
@login_required
def history():
    data = get_user_history(session["user_id"])
    return render_template("history.html", history=data)


@user_bp.route("/history/delete/<int:pred_id>")
@login_required
def history_delete(pred_id):
    delete_prediction(pred_id, session["user_id"])
    flash("Đã xóa lịch sử dự đoán.", "success")
    return redirect(url_for("user_views.history"))