from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

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

info_handler = logging.FileHandler(log_dir / "feature_transformation.log", encoding="utf-8")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

error_handler = logging.FileHandler(error_dir / "feature_transformation_errors.log", encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(info_handler)
logger.addHandler(error_handler)

def log_transform(df, cols=None):
    if cols is None:
        cols = ["Volume", "Volume_MA_7"]
    try:
        for col in cols:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col])
        logger.info(f"Log Transformation tugadi: {cols}")
        print(f"[OK] Log Transformation — {cols}")
        return df
    except Exception as e:
        logger.error(f"Log Transformation xatolik: {e}")
        return df

def robust_scale(df, exclude=None):
    if exclude is None:
        exclude = ["Close", "Ticker", "Currency"]
    try:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logger.info(f"Robust Scaling tugadi: {len(num_cols)} ustun")
        print(f"[OK] Robust Scaling — {len(num_cols)} ustun")
        return df, scaler
    except Exception as e:
        logger.error(f"Robust Scaling xatolik: {e}")
        return df, None

if __name__ == "__main__":
    logger.info("Transformation skript boshlandi")

    file_path = data_dir / "featured_data.csv"
    if not file_path.exists():
        logger.error(f"Fayl topilmadi: {file_path}")
        raise FileNotFoundError(f"{file_path} topilmadi")

    df = pd.read_csv(file_path).dropna()

    print("=" * 50)
    print("TRANSFORMATION BOSHLANDI")
    print(f"Boshlang'ich shape: {df.shape}")
    print("=" * 50)

    df = log_transform(df, cols=["Volume", "Volume_MA_7"])
    df, scaler = robust_scale(df)

    print("\n" + "=" * 50)
    print(f"Yakuniy shape: {df.shape}")
    print("=" * 50)

    save_path = data_dir / "transformed_data.csv"
    df.to_csv(save_path, index=False)

    logger.info(f"Transformed data saqlandi: {save_path}")
    print(f"\n[OK] Saqlandi: {save_path}")
    print(df.head())

    logger.info("Transformation skript tugadi")