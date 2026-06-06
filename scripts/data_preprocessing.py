import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ✅ LOGGING SOZLAMASI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(BASE_DIR, "log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "preprocesser.log"),
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8'
)

logger = logging.getLogger(__name__)

#  1. TOZALA
def tozala(df):
    logger.info("Tozalash boshlandi")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    logger.info("Tozalash tugadi")
    return df


#  2. ENCODING
def encodla(df):
    logger.info("Encoding boshlandi")
    for col in df.columns:
        if df[col].dtype == 'object':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
    logger.info("Encoding tugadi")
    return df


#  3. SCALING
def scale(df):
    logger.info("Scaling boshlandi")
    scaler = MinMaxScaler()
    num_col = df.select_dtypes(include=['int64', 'float64']).columns.drop('Close')
    df[num_col] = scaler.fit_transform(df[num_col])
    logger.info("Scaling tugadi")
    return df


#  MAIN PIPELINE
def preprocess():
    logger.info("Preprocessing boshlandi")

    # ✅ TUZATILDI — file_path to'g'ri
    file_path = os.path.join(BASE_DIR, "Data", "Raw_Data", "all_stocks.csv")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Fayl o'qildi! Qatorlar: {len(df)}, Ustunlar: {len(df.columns)}")

        df = tozala(df)
        df = encodla(df)
        df = scale(df)

        save_dir = os.path.join(BASE_DIR, "Data", "Preprocessed_Data")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "clean_data.csv")
        df.to_csv(save_path, index=False)

        logger.info(f"Fayl saqlandi: {save_path}")
        print("Preprocessing done!")

    except FileNotFoundError:
        logger.error(f"Fayl topilmadi: {file_path}")

    except Exception as e:
        logger.error(f"Xatolik: {e}")


if __name__ == "__main__":
    preprocess()