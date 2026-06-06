import os
import logging
import pandas as pd
import numpy as np

# ─── Yo'llar ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(BASE_DIR, "log")
error_dir = os.path.join(BASE_DIR, "errors")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

# ─── Logging ───────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

info_handler = logging.FileHandler(
    os.path.join(log_dir, "feature_creation.log"),
    encoding='utf-8'
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

error_handler = logging.FileHandler(
    os.path.join(error_dir, "feature_creation_errors.log"),
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))

logger.addHandler(info_handler)
logger.addHandler(error_handler)


def create_features(df):
    logger.info("Feature creation boshlandi")

    try:
        # ─── 1. Moving Average ─────────────────────────
        df["MA_7"]  = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(7).mean())
        df["MA_14"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(14).mean())
        df["MA_30"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(30).mean())
        logger.info("Moving Average yaratildi")

        # ─── 2. Price Change % ─────────────────────────
        df["Price_Change"]   = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change())
        df["Price_Change_7"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(7))
        logger.info("Price Change yaratildi")

        # ─── 3. High-Low Spread ────────────────────────
        df["HL_Spread"]    = df["High"] - df["Low"]
        df["HL_Spread_Pct"] = df["HL_Spread"] / df["Close"]
        logger.info("High-Low Spread yaratildi")

        # ─── 4. Volatility ─────────────────────────────
        df["Volatility_7"]  = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(7).std())
        df["Volatility_14"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(14).std())
        logger.info("Volatility yaratildi")

        # ─── 5. Volume Change ──────────────────────────
        df["Volume_Change"]    = df.groupby("Ticker")["Volume"].transform(lambda x: x.pct_change())
        df["Volume_MA_7"]      = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(7).mean())
        logger.info("Volume Change yaratildi")

        # ─── 6. RSI ────────────────────────────────────
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain  = delta.where(delta > 0, 0).rolling(period).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs    = gain / loss
            return 100 - (100 / (1 + rs))

        df["RSI_14"] = df.groupby("Ticker")["Close"].transform(compute_rsi)
        logger.info("RSI yaratildi")

        # ─── 7. Open-Close farqi ───────────────────────
        df["OC_Diff"]     = df["Close"] - df["Open"]
        df["OC_Diff_Pct"] = df["OC_Diff"] / df["Open"]
        logger.info("Open-Close farqi yaratildi")

        logger.info(f"Feature creation tugadi! Jami ustunlar: {len(df.columns)}")
        print(f"[OK] Jami {len(df.columns)} ta ustun yaratildi!")
        return df

    except Exception as e:
        logger.error(f"Feature creation xatolik: {e}")
        print(f"[FAIL] Xatolik: {e}")
        return None


if __name__ == "__main__":
    logger.info("Creation skript boshlandi")

    # Data yuklash
    file_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
    df = pd.read_csv(file_path)

    # Featurelar yaratish
    df = create_features(df)

    if df is not None:
        # Saqlash
        save_dir = os.path.join(BASE_DIR, "Data", "Engineered_Data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "featured_data.csv")
        df.to_csv(save_path, index=False)
        logger.info(f"Fayl saqlandi: {save_path}")
        print(f"[OK] Fayl saqlandi: {save_path}")
        print(df.head())

    logger.info("Creation skript tugadi")