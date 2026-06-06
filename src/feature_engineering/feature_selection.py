from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parents[2]
log_dir = BASE_DIR / "log"
error_dir = BASE_DIR / "errors"
data_dir = BASE_DIR / "Data" / "Engineered_Data"

log_dir.mkdir(parents=True, exist_ok=True)
error_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

info_handler = logging.FileHandler(log_dir / "feature_selection.log", encoding="utf-8")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

error_handler = logging.FileHandler(error_dir / "feature_selection_errors.log", encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(info_handler)
logger.addHandler(error_handler)

def remove_correlated(X, threshold=0.95):
    logger.info("Korrelyatsiya tekshiruvi boshlandi")
    try:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        X_reduced = X.drop(columns=to_drop)
        logger.info(f"O'chirilgan ustunlar: {to_drop}")
        print(f"[OK] Korrelyatsiya — O'chirildi: {to_drop}")
        print(f"[OK] Qolgan ustunlar: {len(X_reduced.columns)}")
        return X_reduced, to_drop
    except Exception as e:
        logger.error(f"Korrelyatsiya xatolik: {e}")
        return X, []

def select_by_importance(X, y, threshold=0.01):
    logger.info("Feature Importance boshlandi")
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        selected = importance[importance > threshold].index.tolist()
        logger.info(f"Feature Importance tugadi: {len(selected)} feature")
        print("\n[OK] Feature Importance — Top 10:")
        print(importance.sort_values(ascending=False).head(10).to_string())
        print(f"\n[OK] Tanlangan ({len(selected)} ta): {selected}")
        return selected, importance
    except Exception as e:
        logger.error(f"Feature Importance xatolik: {e}")
        return [], None

if __name__ == "__main__":
    logger.info("Selection skript boshlandi")

    file_path = data_dir / "featured_data.csv"
    if not file_path.exists():
        logger.error(f"Fayl topilmadi: {file_path}")
        raise FileNotFoundError(f"{file_path} topilmadi")

    df = pd.read_csv(file_path).dropna()

    if "Close" not in df.columns:
        raise ValueError("Close ustuni topilmadi")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col != "Close"]

    X = df[num_cols]
    y = df["Close"]

    print("=" * 50)
    print("FEATURE SELECTION BOSHLANDI")
    print(f"Boshlang'ich ustunlar: {len(X.columns)}")
    print("=" * 50)

    X_reduced, dropped = remove_correlated(X, threshold=0.95)
    imp_features, imp_scores = select_by_importance(X_reduced, y, threshold=0.01)

    if imp_features:
        save_path = data_dir / "selected_features.csv"
        selected_df = df[imp_features + ["Close"]]
        selected_df.to_csv(save_path, index=False)

        logger.info(f"Saqlandi: {save_path}")
        print(f"\n[OK] Saqlandi: {save_path}")
        print("SAVE PATH:", save_path)
        print("EXISTS AFTER SAVE:", save_path.exists())
    else:
        logger.warning("Tanlangan feature topilmadi")
        print("[WARN] Tanlangan feature topilmadi")

    print("\n" + "=" * 50)
    print("FEATURE SELECTION TUGADI!")
    print("=" * 50)

    logger.info("Selection skript tugadi")