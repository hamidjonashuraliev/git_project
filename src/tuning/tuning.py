from pathlib import Path
import os
import logging
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[2]

log_dir    = BASE_DIR / "log"
error_dir  = BASE_DIR / "errors"
result_dir = BASE_DIR / "results"
data_dir   = BASE_DIR / "Data" / "Preprocessed_Data"
model_dir  = BASE_DIR / "models"

log_dir.mkdir(exist_ok=True)
error_dir.mkdir(exist_ok=True)
result_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

info_handler = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(error_dir / "training_errors.log", mode="a", encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)


# ─── Data yuklash ──────────────────────────────────────
def load_data():
    file_path = data_dir / "clean_data.csv"
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data yuklandi! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Fayl topilmadi: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Data yuklash xatosi: {e}")
        return None


# ─── Train/Test ajratish ───────────────────────────────
def split_data(df):
    X = df.drop(columns=["Close"])
    y = df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─── RandomForest Tuning ───────────────────────────────
def tune_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    param_dist = {
        "n_estimators":     [50, 100, 150, 200, 300],
        "max_depth":        [None, 5, 10, 20, 30],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10, scoring="r2", cv=3,
        random_state=42, n_jobs=-1,
        verbose=1, return_train_score=True
    )
    search.fit(X_train, y_train)
    logger.info(f"RF best params: {search.best_params_}")
    logger.info(f"RF best CV R2: {search.best_score_:.4f}")
    return search


# ─── XGBoost Tuning ────────────────────────────────────
def tune_xgboost(X_train, y_train):
    model = XGBRegressor(random_state=42, objective="reg:squarederror")
    param_dist = {
        "n_estimators":     [50, 100, 150, 200, 300],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.2, 0.5],
    }
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10, scoring="r2", cv=3,
        random_state=42, n_jobs=-1,
        verbose=1, return_train_score=True
    )
    search.fit(X_train, y_train)
    logger.info(f"XGB best params: {search.best_params_}")
    logger.info(f"XGB best CV R2: {search.best_score_:.4f}")
    return search


# ─── Test natijalar ────────────────────────────────────
def evaluate_on_test(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"{model_name} TEST R2: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")
    return r2, mae, mse


# ─── Jadval chiqarish ──────────────────────────────────
def print_results_table(rows):
    print("\n" + "=" * 75)
    print(f"{'MODEL':<20} {'CV R2':>8} {'TEST R2':>8} {'MAE':>10} {'MSE':>12}")
    print("-" * 75)
    for row in rows:
        print(
            f"{row['model']:<20} "
            f"{row['cv_r2']:>8.4f} "
            f"{row['test_r2']:>8.4f} "
            f"{row['test_mae']:>10.4f} "
            f"{row['test_mse']:>12.4f}"
        )
    print("=" * 75)


# ─── Test data saqlash ─────────────────────────────────
def save_test_data(X_test, y_test):
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)
    logger.info("X_test va y_test saqlandi")


# ─── Natijalar jadvali saqlash ─────────────────────────
def save_results_table(rows):
    df = pd.DataFrame(rows)
    out_path = result_dir / "tuning_results.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Natijalar saqlandi: {out_path}")
    print(f"\n[OK] Natijalar saqlandi: {out_path}")


# ─── Eng yaxshi modelni saqlash ───────────────────────
def save_best_model(model, name):
    model_path = model_dir / "best_algorithm.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Eng yaxshi model saqlandi: {name} → {model_path}")
    print(f"\n[OK] Eng yaxshi model: {name}")
    print(f"[OK] Saqlandi: {model_path}")


# ─── Main ──────────────────────────────────────────────
def main():
    try:
        logger.info("Tuning boshlandi")
        print("\n" + "=" * 75)
        print("HYPERPARAMETER TUNING BOSHLANDI")
        print("=" * 75)

        df = load_data()
        if df is None:
            return

        if "Close" not in df.columns:
            logger.error("Close ustuni topilmadi.")
            return

        X_train, X_test, y_train, y_test = split_data(df)
        save_test_data(X_test, y_test)

        results_rows = []

        # RandomForest
        print("\nRandomForest tuning...")
        rf_search = tune_random_forest(X_train, y_train)
        rf_best   = rf_search.best_estimator_
        rf_r2, rf_mae, rf_mse = evaluate_on_test(rf_best, X_test, y_test, "RandomForest")
        results_rows.append({
            "model":       "RandomForest",
            "best_params": str(rf_search.best_params_),
            "cv_r2":       rf_search.best_score_,
            "test_r2":     rf_r2,
            "test_mae":    rf_mae,
            "test_mse":    rf_mse,
        })

        # XGBoost
        print("\nXGBoost tuning...")
        xgb_search = tune_xgboost(X_train, y_train)
        xgb_best   = xgb_search.best_estimator_
        xgb_r2, xgb_mae, xgb_mse = evaluate_on_test(xgb_best, X_test, y_test, "XGBoost")
        results_rows.append({
            "model":       "XGBoost",
            "best_params": str(xgb_search.best_params_),
            "cv_r2":       xgb_search.best_score_,
            "test_r2":     xgb_r2,
            "test_mae":    xgb_mae,
            "test_mse":    xgb_mse,
        })

        # Jadval
        print_results_table(results_rows)
        save_results_table(results_rows)

        # Eng yaxshisi
        if xgb_r2 >= rf_r2:
            best_model, best_name = xgb_best, "XGBoost"
        else:
            best_model, best_name = rf_best, "RandomForest"

        save_best_model(best_model, best_name)
        logger.info("Tuning tugadi")

    except Exception as e:
        logger.exception(f"Main xatolik: {e}")


if __name__ == "__main__":
    main()