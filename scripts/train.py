import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# ✅ LOGGING
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(BASE_DIR, "log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "train.log"),
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


# ─── Data yuklash ──────────────────────────────────────
def load_data():
    file_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data yuklandi! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Fayl topilmadi: {file_path}")
    except Exception as e:
        logger.error(f"Xatolik: {e}")


# ─── Train/Test ajratish ───────────────────────────────
def split_data(df):
    X = df.drop(columns=["Close"])
    y = df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─── Modellar ──────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":          XGBRegressor(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        try:
            logger.info(f"{name} o'qitilmoqda...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2  = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results[name] = {"model": model, "r2": r2, "mae": mae}
            logger.info(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")
            print(f"{name:20} | R2: {r2:.4f} | MAE: {mae:.4f}")

        except Exception as e:
            logger.error(f"{name} xatolik: {e}")

    return results


# ─── Eng yaxshi modelni saqlash ───────────────────────
def save_best_model(results):
    best_name = max(results, key=lambda x: results[x]["r2"])
    best_model = results[best_name]["model"]

    logger.info(f"Eng yaxshi model: {best_name} — R2: {results[best_name]['r2']:.4f}")
    print(f"\nEng yaxshi model: {best_name}")
    print(f"R2:  {results[best_name]['r2']:.4f}")
    print(f"MAE: {results[best_name]['mae']:.4f}")

    save_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pkl")

    joblib.dump(best_model, save_path)
    logger.info(f"Model saqlandi: {save_path}")
    print(f"Model saqlandi: {save_path}")


# ─── Ishga tushirish ───────────────────────────────────
if __name__ == "__main__":
    logger.info("Train skript boshlandi")

    df = load_data()

    if df is not None:
        X_train, X_test, y_train, y_test = split_data(df)
        results = train_models(X_train, X_test, y_train, y_test)
        save_best_model(results)

    logger.info("Train skript tugadi")