import os
import logging
import pandas as pd
import yfinance as yf

# ─── Yo'llar ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(BASE_DIR, "log")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "data_loader.log")

# ─── Logging sozlamasi ─────────────────────────────────
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


# ─── Yuklash va saqlash ────────────────────────────────
def load_and_save(tickers, period="1y"):
    logger.info("Data yuklash boshlandi")

    save_dir = os.path.join(BASE_DIR, "Data", "Raw_Data")
    os.makedirs(save_dir, exist_ok=True)

    all_data = []

    for ticker in tickers:
        try:
            logger.info(f"{ticker} yuklanmoqda...")
            df = yf.Ticker(ticker).history(period=period)

            if df.empty:
                logger.warning(f"{ticker} uchun data topilmadi!")
                continue

            df["Ticker"] = ticker
            df["Currency"] = "USD"
            all_data.append(df)

            path = os.path.join(save_dir, f"{ticker}.csv")
            df.to_csv(path)
            logger.info(f"{ticker} saqlandi!")

        except Exception as e:
            logger.error(f"{ticker} xatolik: {e}")

    final_df = pd.concat(all_data)
    final_path = os.path.join(save_dir, "all_stocks.csv")
    final_df.to_csv(final_path)
    logger.info(f"Barcha data saqlandi: {final_path}")

    return final_df


# ─── O'qish ───────────────────────────────────────────
def load_data():
    logger.info("load_data() boshlandi")
    file_path = os.path.join(BASE_DIR, "Data", "Raw_Data", "all_stocks.csv")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Fayl o'qildi! Qatorlar: {len(df)}, Ustunlar: {len(df.columns)}")
        return df

    except FileNotFoundError:
        logger.error(f"Fayl topilmadi: {file_path}")

    except Exception as e:
        logger.error(f"Kutilmagan xatolik: {e}")


# ─── Ishga tushirish ───────────────────────────────────
if __name__ == "__main__":
    logger.info("Skript boshlandi")

    tickers = [
        "AAPL",   # Apple
        "TSLA",   # Tesla
        "MSFT",   # Microsoft
        "GOOGL",  # Google
        "AMZN",   # Amazon
        "META",   # Meta
        "NVDA",   # Nvidia
        "NFLX",   # Netflix
        "INTC",   # Intel
        "AMD",    # AMD
        "TSM",    # TSMC
        "BABA",   # Alibaba
    ]

    load_and_save(tickers, period="1y")

    df = load_data()
    if df is not None:
        print(df.head())
        print(df.info())

    logger.info("Skript tugadi")