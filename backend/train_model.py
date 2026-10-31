"""
train_model.py — Huấn luyện mô hình dự đoán giá nhà (2-component).

Kiến trúc:
  • Baseline: đường cong lũy thừa log(ppm) = a + b*log(area) (calibrate từ data)
  • Residual model: học phần chênh lệch theo location/ward/rooms/toilets

Đảm bảo logic đơn điệu: nhà lớn hơn (cùng thông số) → giá cao hơn.
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.location_service import (
    normalize_city, normalize_ward, build_location_catalog,
    save_location_catalog, clear_location_cache
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR     = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "database" / "dataset.csv"
MODEL_PATH   = ROOT_DIR / "models" / "trained_model.pkl"
REPORT_PATH  = ROOT_DIR / "models" / "model_report.txt"
COMPARISON_PATH = ROOT_DIR / "models" / "model_comparison.csv"

# Ngưỡng phân biệt giá bán vs giá thuê trong dataset
SALE_THRESHOLD = 500_000_000   # >= 500 triệu → bán

# Feature columns (NO area — area handled by baseline curve)
FEATURE_COLUMNS   = ["location", "ward", "rooms", "toilets"]
CATEGORICAL_FEATS = ["location", "ward"]
NUMERIC_FEATS     = ["rooms", "toilets"]


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def clean_dataset(df: pd.DataFrame, mode: str = "sale") -> pd.DataFrame:
    """Chuẩn hóa dataset cũ/mới; chỉ giữ tin bán nhà tại TP.HCM.

    Không dùng ngưỡng giá 500 triệu để phân biệt bán/thuê vì nhà bán giá thấp
    vẫn có thể hợp lệ và giá thuê có thể vượt ngưỡng này.
    """
    df = df.copy()
    if mode != "sale":
        raise ValueError("Dataset cập nhật hiện chỉ gồm tin BÁN nhà TP.HCM; không thể train chế độ giá thuê.")

    # Chuẩn hóa tên cột từ bộ Tinix/Hugging Face sang schema cũ của ứng dụng.
    aliases = {
        "province_name": "location", "ward_name": "ward",
        "bedroom_count": "rooms", "bathroom_count": "toilets",
        "name": "title", "description": "description",
    }
    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]

    for col in ("location", "ward", "rooms", "toilets", "title", "description"):
        if col not in df.columns:
            df[col] = np.nan

    # Chỉ sử dụng dữ liệu TP. Hồ Chí Minh (kể cả các cách ghi tên tương đương).
    city_text = df["location"].fillna("").astype(str).str.lower()
    hcm_mask = city_text.str.contains(r"hồ\s*chí\s*minh|ho\s*chi\s*minh|tp\.?\s*hcm|tphcm", regex=True)
    df = df.loc[hcm_mask].copy()

    # Dự án này định giá nhà riêng/nhà phố; loại đất và căn hộ khỏi tập train.
    if "property_type_name" in df.columns:
        allowed = df["property_type_name"].fillna("").astype(str).str.lower().str.strip()
        allowed_mask = allowed.isin({"nhà", "nha", "biệt thự/nhà liền kề", "biet thu/nha lien ke",
                                     "biệt thự / nhà liền kề", "biet thu / nha lien ke",
                                     "nhà mặt phố", "nha mat pho"})
        df = df.loc[allowed_mask].copy()

    # Lọc tin cho thuê dựa trên nội dung, không dùng ngưỡng giá cố định.
    title_text = df["title"].fillna("").astype(str).str.lower()
    all_text = (title_text + " " + df["description"].fillna("").astype(str).str.lower())
    rent_pattern = r"cho thuê|cần thuê|tìm thuê|thuê nhà|thuê nguyên căn|\bcho thue\b|\bcan thue\b|\btim thue\b|\bthue nha\b|/tháng|/thang|mỗi tháng|moi thang|per month"
    sale_pattern = r"bán|cần bán|chuyển nhượng|sang nhượng|ban nha|can ban|chuyen nhuong"
    rental_title = title_text.str.contains(rent_pattern, regex=True, na=False)
    explicit_sale = all_text.str.contains(sale_pattern, regex=True, na=False)
    # Tin có tiêu đề cho thuê luôn bị loại; tin bán có nhắc hợp đồng thuê trong mô tả vẫn được giữ.
    df = df.loc[~rental_title & (explicit_sale | ~all_text.str.contains(rent_pattern, regex=True, na=False))].copy()

    # Xóa trùng trước khi xử lý giá.
    df = df.drop_duplicates()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["price", "area"])
    df = df[(df["price"] >= 100_000_000) & (df["price"] <= 200_000_000_000)]
    df = df[(df["area"] >= 10) & (df["area"] <= 1_000)]
    df["price_per_m2"] = df["price"] / df["area"]
    df = df[df["price_per_m2"].between(5_000_000, 1_000_000_000)]

    # Loại phần đuôi bất thường theo log giá/m² (0,5% mỗi phía), tránh IQR
    # quá gắt làm mất các căn nhà ở phân khúc cao/thấp hợp lệ.
    log_ppm = np.log(df["price_per_m2"])
    if len(df) >= 100:
        low, high = log_ppm.quantile([0.005, 0.995])
        df = df[log_ppm.between(low, high)].copy()

    df["location"] = df["location"].apply(normalize_city)
    df["ward"] = df["ward"].apply(normalize_ward)
    df = df[(df["location"] != "") & (df["ward"] != "")]
    # Chuẩn hóa toàn bộ dữ liệu về một tên thành phố duy nhất.
    df["location"] = "Tp Hồ Chí Minh"

    for col in ("rooms", "toilets"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df.reset_index(drop=True)


def get_estimator(algorithm: str, n_estimators: int, max_depth: int):
    if algorithm == "linear_regression":
        return LinearRegression()
    elif algorithm == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=5, random_state=42,
        )
    elif algorithm == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth < 20 else None,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        )
    elif algorithm == "ridge":
        return Ridge(alpha=10.0)
    else:
        return GradientBoostingRegressor(n_estimators=300, random_state=42)


def build_pipeline(estimator) -> Pipeline:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATS),
        ("num", num_pipeline, NUMERIC_FEATS),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def validate_model_bundle(bundle) -> None:
    required_keys = {
        "model", "slope", "intercept", "feats", "algorithm", "mode", "accuracy",
    }
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("Saved model has an invalid bundle structure.")

    model = bundle["model"]
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Saved model bundle does not contain a predictor.")

    # Validate location_catalog if present in bundle
    if "location_catalog" in bundle:
        catalog = bundle["location_catalog"]
        if not isinstance(catalog, dict) or not catalog:
            raise ValueError("Saved model bundle contains an invalid or empty location_catalog.")

    sample = pd.DataFrame([{
        "location": "unknown",
        "ward": "unknown",
        "rooms": 0.0,
        "toilets": 0.0,
    }])
    prediction = model.predict(sample)
    if len(prediction) != 1 or not np.isfinite(prediction[0]):
        raise ValueError("Saved model failed its prediction validation.")

    for key in ("slope", "intercept", "accuracy"):
        if not np.isfinite(float(bundle[key])):
            raise ValueError(f"Saved model bundle contains an invalid {key}.")


def save_model_bundle(bundle: dict, model_path: Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    temporary_catalog_path = None
    stage = "create temporary model file"
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{model_path.name}.",
            suffix=".tmp",
            dir=model_path.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)

        stage = "serialize temporary model"
        joblib.dump(bundle, temporary_path)

        stage = "reload and validate temporary model"
        validate_model_bundle(joblib.load(temporary_path))

        # Lưu danh mục địa điểm ra file tạm nếu có trong bundle
        if "location_catalog" in bundle:
            cat_fd, cat_name = tempfile.mkstemp(
                prefix=".location_catalog.",
                suffix=".tmp",
                dir=model_path.parent,
            )
            os.close(cat_fd)
            temporary_catalog_path = Path(cat_name)
            with open(temporary_catalog_path, "w", encoding="utf-8") as f:
                json.dump(bundle["location_catalog"], f, ensure_ascii=False, indent=2)

        stage = "replace active model"
        os.replace(temporary_path, model_path)
        temporary_path = None

        if temporary_catalog_path is not None:
            stage = "replace active location catalog"
            catalog_target = model_path.parent / "location_catalog.json"
            os.replace(temporary_catalog_path, catalog_target)
            temporary_catalog_path = None

        stage = "reload and validate saved model"
        validate_model_bundle(joblib.load(model_path))
        clear_location_cache()
    except OSError as exc:
        parent_exists = model_path.parent.is_dir()
        target_exists = model_path.exists()
        raise OSError(
            f"Failed to {stage}: target={str(model_path)!r}, "
            f"parent_exists={parent_exists}, target_exists={target_exists}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        if temporary_catalog_path is not None and temporary_catalog_path.exists():
            temporary_catalog_path.unlink(missing_ok=True)


def clean_and_split(df_raw: pd.DataFrame, mode: str = "sale") -> pd.DataFrame:
    """Alias cho app.py."""
    return clean_dataset(df_raw, mode=mode)


def compare_models(test_size: float = 0.2, n_estimators: int = 300,
                   max_depth: int = 4, mode: str = "sale",
                   output_path: Path = COMPARISON_PATH) -> pd.DataFrame:
    """So sánh mô hình trên cùng tập test theo thời gian (nếu có ngày đăng)."""
    df_raw = load_dataset()
    df_clean = clean_dataset(df_raw, mode=mode)
    print(f"[compare] TP.HCM | Raw: {len(df_raw)} -> Clean: {len(df_clean)}")
    if len(df_clean) < 50:
        raise ValueError(f"Dữ liệu sau làm sạch chỉ còn {len(df_clean)} dòng.")

    train_df, test_df, split_method = _split_train_test(df_clean, test_size)
    slope, intercept = _fit_baseline(train_df)
    y_train = np.log(train_df["price"] / train_df["area"]) - (
        intercept + slope * np.log(train_df["area"]))
    X_train, X_test = train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]
    log_area_test = np.log(test_df["area"].to_numpy())
    area_test = test_df["area"].to_numpy()
    actual_prices = test_df["price"].to_numpy()

    algorithms = (
        ("linear_regression", LinearRegression()),
        ("random_forest", get_estimator("random_forest", n_estimators, max_depth)),
        ("gradient_boosting", get_estimator("gradient_boosting", n_estimators, max_depth)),
    )
    results = []
    for algorithm, estimator in algorithms:
        pipeline = build_pipeline(estimator)
        print(f"[compare] Training {algorithm} on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)
        predicted_residuals = pipeline.predict(X_test)
        predicted_prices = np.exp(intercept + slope * log_area_test + predicted_residuals) * area_test
        results.append({
            "algorithm": algorithm,
            "mae": mean_absolute_error(actual_prices, predicted_prices),
            "rmse": float(mean_squared_error(actual_prices, predicted_prices) ** 0.5),
            "r2": r2_score(actual_prices, predicted_prices),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "split_method": split_method,
        })

    comparison = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False, formatters={
        "mae": "{:,.0f}".format, "rmse": "{:,.0f}".format, "r2": "{:.4f}".format,
    }))
    print(f"[compare] Results saved to {output_path}")
    return comparison


def _split_train_test(df: pd.DataFrame, test_size: float):
    """Dùng tập test theo thời gian nếu dataset có ngày đăng; fallback random."""
    if "published_at" in df.columns:
        dates = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        valid = dates.notna()
        if valid.sum() >= 50:
            ordered = df.loc[valid].assign(_date=dates[valid]).sort_values("_date")
            split_at = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_size))))
            return ordered.iloc[:split_at].drop(columns=["_date"]), ordered.iloc[split_at:].drop(columns=["_date"]), "time"
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df, "random"


def _fit_baseline(df: pd.DataFrame):
    log_area = np.log(df["area"].to_numpy())
    log_ppm = np.log((df["price"] / df["area"]).to_numpy())
    slope, intercept, *_ = stats.linregress(log_area, log_ppm)
    return float(slope), float(intercept)


def train(algorithm: str = "gradient_boosting", test_size: float = 0.2,
          n_estimators: int = 300, max_depth: int = 4,
          mode: str = "sale") -> dict:
    """Đánh giá trên holdout rồi fit model cuối cùng bằng toàn bộ dữ liệu sạch."""
    df_raw = load_dataset()
    df_clean = clean_dataset(df_raw, mode=mode)
    n_raw, n_clean = len(df_raw), len(df_clean)
    print(f"[train] TP.HCM | Raw: {n_raw} -> Clean ({mode}): {n_clean}")
    if n_clean < 50:
        raise ValueError(f"Dữ liệu TP.HCM sau làm sạch chỉ còn {n_clean} dòng; cần ít nhất 50 dòng.")

    train_df, test_df, split_method = _split_train_test(df_clean, test_size)
    if len(train_df) < 25 or len(test_df) < 10:
        raise ValueError("Không đủ dữ liệu cho tập train/test sau khi chia.")

    # Đánh giá: baseline chỉ fit trên train để tránh rò rỉ dữ liệu test.
    eval_slope, eval_intercept = _fit_baseline(train_df)
    y_train = np.log(train_df["price"] / train_df["area"]) - (
        eval_intercept + eval_slope * np.log(train_df["area"]))
    X_train, X_test = train_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]
    estimator = get_estimator(algorithm, n_estimators, max_depth)
    eval_pipeline = build_pipeline(estimator)
    print(f"[train] Training {algorithm} on {len(X_train)} samples; test={len(X_test)} ({split_method})")
    eval_pipeline.fit(X_train, y_train)

    y_pred_resid = eval_pipeline.predict(X_test)
    test_log_area = np.log(test_df["area"].to_numpy())
    predicted_prices = np.exp(
        eval_intercept + eval_slope * test_log_area + y_pred_resid
    ) * test_df["area"].to_numpy()
    actual_prices = test_df["price"].to_numpy()
    mae_price = float(mean_absolute_error(actual_prices, predicted_prices))
    rmse_price = float(mean_squared_error(actual_prices, predicted_prices) ** 0.5)
    r2_price = float(r2_score(actual_prices, predicted_prices))
    r2_log = float(r2_score(np.log(actual_prices), np.log(np.maximum(predicted_prices, 1))))
    # Giữ key 'accuracy' để tương thích giao diện hiện tại; đây là R² log quy đổi,
    # không phải accuracy của bài toán phân loại.
    accuracy = round(max(0.0, min(r2_log, 1.0)) * 100, 1)
    residual_r2 = float(r2_score(y_train, eval_pipeline.predict(X_train)))

    # Fit mô hình production trên toàn bộ dữ liệu sạch sau khi đã đánh giá.
    final_slope, final_intercept = _fit_baseline(df_clean)
    final_residual = np.log(df_clean["price"] / df_clean["area"]) - (
        final_intercept + final_slope * np.log(df_clean["area"]))
    final_pipeline = build_pipeline(get_estimator(algorithm, n_estimators, max_depth))
    final_pipeline.fit(df_clean[FEATURE_COLUMNS], final_residual)

    wards = sorted(w for w in df_clean["ward"].dropna().astype(str).unique() if w.strip())
    location_catalog = {"Tp Hồ Chí Minh": wards} if wards else {}
    if not location_catalog:
        raise ValueError("Dataset không có danh mục phường/xã hợp lệ.")

    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bundle = {
        "model": final_pipeline,
        "slope": float(final_slope),
        "intercept": float(final_intercept),
        "feats": FEATURE_COLUMNS,
        "algorithm": algorithm,
        "mode": mode,
        "accuracy": accuracy,
        "location_catalog": location_catalog,
        "trained_at": trained_at,
        "training_region": "Tp Hồ Chí Minh",
        "training_rows": int(n_clean),
        "evaluation_split": split_method,
        "evaluation_r2_log": r2_log,
        "evaluation_mae_vnd": mae_price,
    }
    save_model_bundle(bundle, MODEL_PATH)
    df_clean.to_csv(MODEL_PATH.parent / "trained_data.csv", index=False, encoding="utf-8-sig")

    report = "\n".join([
        "=" * 58,
        "  HOUSE PRICE AI — BÁO CÁO HUẤN LUYỆN",
        "=" * 58,
        f"Thời gian             : {trained_at}",
        f"Phạm vi dữ liệu       : TP. Hồ Chí Minh",
        f"Thuật toán            : {algorithm}",
        f"Chế độ                : {'Giá bán' if mode == 'sale' else 'Giá thuê'}",
        f"Phương pháp chia      : {split_method} (test {int(test_size * 100)}%)",
        f"Dữ liệu gốc           : {n_raw:,} bản ghi",
        f"Dữ liệu sau làm sạch  : {n_clean:,} bản ghi",
        f"Tập train             : {len(train_df):,} mẫu",
        f"Tập test              : {len(test_df):,} mẫu",
        f"Số phường/xã          : {sum(len(w) for w in location_catalog.values()):,}",
        "-" * 58,
        f"R² giá log            : {r2_log:.4f}",
        f"R² giá thực           : {r2_price:.4f}",
        f"Điểm R² log quy đổi   : {accuracy}% (không phải accuracy phân loại)",
        f"MAE                   : {mae_price / 1_000_000:.1f} triệu VNĐ",
        f"RMSE                  : {rmse_price / 1_000_000:.1f} triệu VNĐ",
        f"R² residual trên train: {residual_r2:.4f}",
        f"Baseline slope        : {final_slope:.4f}",
        f"Baseline intercept    : {final_intercept:.4f}",
        "=" * 58,
    ])
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    return {
        "algorithm": algorithm, "mode": mode, "r2": round(r2_log, 4),
        "mae": round(mae_price, 0), "rmse": round(rmse_price, 0),
        "accuracy": accuracy, "train_rows": len(train_df), "test_rows": len(test_df),
        "original_rows": n_raw, "cleaned_rows": n_clean, "trained_at": trained_at,
        "location_catalog": location_catalog, "baseline_slope": round(final_slope, 4),
        "baseline_intercept": round(final_intercept, 4), "evaluation_split": split_method,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["linear_regression", "random_forest", "gradient_boosting"],
        default="linear_regression"
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true"
    )

    args = parser.parse_args()

    if not args.skip_compare:
        compare_models()

    train(algorithm=args.algorithm, mode="sale")