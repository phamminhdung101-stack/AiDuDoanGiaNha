"""
train_model.py — Huấn luyện mô hình dự đoán giá nhà (2-component).

Kiến trúc:
  • Baseline: đường cong lũy thừa log(ppm) = a + b*log(area) (calibrate từ data)
  • Residual model: học phần chênh lệch theo location/ward/rooms/toilets

Đảm bảo logic đơn điệu: nhà lớn hơn (cùng thông số) → giá cao hơn.
"""
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR     = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "database" / "dataset.csv"
MODEL_PATH   = ROOT_DIR / "models" / "trained_model.pkl"
REPORT_PATH  = ROOT_DIR / "models" / "model_report.txt"

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
    df = df.copy()
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Convert types and drop missing critical values
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area"]  = pd.to_numeric(df["area"],  errors="coerce")
    df = df.dropna(subset=["price", "area"])
    df = df[(df["price"] > 0) & (df["area"] > 0)]

    # 3. Create price_per_m2 feature
    df["price_per_m2"] = df["price"] / df["area"]

    # 4. Outlier removal using IQR
    Q1_ppm = df['price_per_m2'].quantile(0.25)
    Q3_ppm = df['price_per_m2'].quantile(0.75)
    IQR_ppm = Q3_ppm - Q1_ppm
    df = df[(df['price_per_m2'] >= Q1_ppm - 1.5 * IQR_ppm) & 
            (df['price_per_m2'] <= Q3_ppm + 1.5 * IQR_ppm)]
            
    Q1_area = df['area'].quantile(0.25)
    Q3_area = df['area'].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    df = df[(df['area'] >= Q1_area - 1.5 * IQR_area) & 
            (df['area'] <= Q3_area + 1.5 * IQR_area)]

    # 5. Missing categorical values
    for col in CATEGORICAL_FEATS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "unknown"
        
    # Note: Numeric missing values (rooms, toilets) will be handled by SimpleImputer in the pipeline.
    for col in ["rooms", "toilets"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    return df


def get_estimator(algorithm: str, n_estimators: int, max_depth: int):
    if algorithm == "gradient_boosting":
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


def clean_and_split(df_raw: pd.DataFrame, mode: str = "sale") -> pd.DataFrame:
    """Alias cho app.py."""
    return clean_dataset(df_raw, mode=mode)


def train(algorithm: str = "gradient_boosting", test_size: float = 0.2,
          n_estimators: int = 300, max_depth: int = 4,
          mode: str = "sale") -> dict:
    """
    Train 2-component model và lưu bundle dict.
    Returns metrics dict.
    """
    df_raw   = load_dataset()
    df_clean = clean_dataset(df_raw, mode=mode)

    n_raw, n_clean = len(df_raw), len(df_clean)
    print(f"[train] Raw: {n_raw}  →  Clean ({mode}): {n_clean}")

    if n_clean < 50:
        raise ValueError(f"Dữ liệu sau làm sạch chỉ còn {n_clean} dòng.")

    df_clean["price_per_m2"] = df_clean["price"] / df_clean["area"]
    df_clean["log_ppm"]      = np.log(df_clean["price_per_m2"])
    df_clean["log_area"]     = np.log(df_clean["area"])

    # ── Bước 1: Fit đường cong baseline area ──────────────────────────────
    slope, intercept, r_val, *_ = stats.linregress(
        df_clean["log_area"], df_clean["log_ppm"]
    )
    print(f"[train] Baseline: log(ppm) = {intercept:.3f} + {slope:.3f}*log(area)")
    print(f"[train] Exponent total price: area^{1+slope:.3f}  ({'monotone ✓' if 1+slope > 0 else 'NON-monotone ✗'})")

    # ── Bước 2: Tính residual & train quality model ────────────────────────
    df_clean["log_ppm_baseline"] = intercept + slope * df_clean["log_area"]
    df_clean["log_ppm_residual"] = df_clean["log_ppm"] - df_clean["log_ppm_baseline"]

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean["log_ppm_residual"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATS),
        ("num", num_pipeline, NUMERIC_FEATS),
    ])
    estimator = get_estimator(algorithm, n_estimators, max_depth)
    pipeline  = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    print(f"[train] Training residual model ({algorithm}) on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)

    # ── Metrics trên log_ppm_residual ─────────────────────────────────────
    y_pred   = pipeline.predict(X_test)
    r2       = r2_score(y_test, y_pred)
    mae_log  = mean_absolute_error(y_test, y_pred)

    # Metrics trên giá thực (VNĐ) để dễ hiểu
    def full_predict(row):
        log_ppm_base = intercept + slope * np.log(max(row["area"], 1.0))
        X_q = pd.DataFrame([{c: row[c] for c in FEATURE_COLUMNS}])
        log_ppm_q = float(pipeline.predict(X_q)[0])
        return np.exp(log_ppm_base + log_ppm_q) * row["area"]

    test_idx = X_test.index
    y_price_actual = df_clean.loc[test_idx, "price"].values
    y_price_pred   = np.array([
        full_predict(df_clean.loc[i]) for i in test_idx
    ])
    mae_price  = mean_absolute_error(y_price_actual, y_price_pred)
    rmse_price = float(mean_squared_error(y_price_actual, y_price_pred) ** 0.5)
    r2_price   = r2_score(y_price_actual, y_price_pred)
    r2_log     = r2_score(np.log(y_price_actual), np.log(y_price_pred))
    accuracy   = round(max(0.0, min(r2_log, 1.0)) * 100, 1)

    # CV trên residual model
    cv        = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                scoring="r2", n_jobs=-1)
    cv_mean   = round(float(cv_scores.mean()), 4)
    cv_std    = round(float(cv_scores.std()),  4)

    # ── Lưu bundle ────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":     pipeline,
        "slope":     float(slope),
        "intercept": float(intercept),
        "feats":     FEATURE_COLUMNS,
        "algorithm": algorithm,
        "mode":      mode,
        "accuracy":  accuracy,
    }
    joblib.dump(bundle, MODEL_PATH)

    # Lưu trained_data.csv
    df_clean.to_csv(MODEL_PATH.parent / "trained_data.csv", index=False, encoding="utf-8-sig")

    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = "\n".join([
        "=" * 50,
        "  HOUSE PRICE AI — BÁO CÁO HUẤN LUYỆN",
        "=" * 50,
        f"Thời gian      : {trained_at}",
        f"Thuật toán     : {algorithm} (residual model)",
        f"Chế độ         : {'Giá bán' if mode == 'sale' else 'Giá thuê'}",
        f"n_estimators   : {n_estimators}",
        f"max_depth      : {max_depth}",
        f"Test size      : {int(test_size * 100)}%",
        "-" * 50,
        "BASELINE CURVE",
        f"  slope        : {slope:.4f}",
        f"  intercept    : {intercept:.4f}",
        f"  price ∝ area^{1+slope:.3f}  (monotone tăng)",
        "-" * 50,
        f"Dữ liệu gốc    : {n_raw} bản ghi",
        f"Dữ liệu dùng   : {n_clean} bản ghi",
        f"Tập train      : {len(X_train)} mẫu",
        f"Tập test       : {len(X_test)} mẫu",
        "-" * 50,
        f"R2 (log)       : {r2_log:.4f}",
        f"Accuracy       : {accuracy}%",
        f"MAE            : {mae_price/1_000_000:.1f} triệu VNĐ",
        f"RMSE           : {rmse_price/1_000_000:.1f} triệu VNĐ",
        f"CV R2 (resid.) : {cv_mean} ± {cv_std} (5-fold)",
        "=" * 50,
    ])
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    return {
        "algorithm": algorithm, "mode": mode,
        "r2": round(r2_log, 4), "mae": round(mae_price, 0),
        "rmse": round(rmse_price, 0), "accuracy": accuracy,
        "cv_r2": cv_mean, "cv_std": cv_std,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "original_rows": n_raw, "cleaned_rows": n_clean,
        "trained_at": trained_at,
        "baseline_slope": round(slope, 4),
        "baseline_intercept": round(intercept, 4),
    }


if __name__ == "__main__":
    result = train(algorithm="gradient_boosting", mode="sale")
