import os
import logging
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(BASE_DIR, "log")
error_dir = os.path.join(BASE_DIR, "errors")
result_dir = os.path.join(BASE_DIR, "results")
data_dir = os.path.join(BASE_DIR, "Data", "Preprocessed_Data")
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

info_handler = logging.FileHandler(
    os.path.join(log_dir, "predict.log"),
    mode="a",
    encoding="utf-8"
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(
    os.path.join(error_dir, "predict_errors.log"),
    mode="a",
    encoding="utf-8"
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(stream_handler)

def load_model():
    model_path = os.path.join(model_dir, "best_algorithm.pkl")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model yuklandi: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Model yuklash xatosi: {e}")
        return None

def load_data():
    file_path = os.path.join(data_dir, "clean_data.csv")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data yuklandi: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Data yuklash xatosi: {e}")
        return None

def predict_model():
    try:
        model = load_model()
        if model is None:
            return

        df = load_data()
        if df is None:
            return

        if "Close" in df.columns:
            X = df.drop(columns=["Close"])
        else:
            X = df

        predictions = model.predict(X)

        output = X.copy()
        output["prediction"] = predictions

        output_path = os.path.join(result_dir, "predictions.csv")
        output.to_csv(output_path, index=False)

        logger.info(f"Predictions saqlandi: {output_path}")
        print("Prediction tugadi.")
    except Exception as e:
        logger.error(f"Predict funksiyada xatolik: {e}")

if __name__ == "__main__":
    predict_model()