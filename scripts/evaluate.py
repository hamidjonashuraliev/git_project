import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    os.path.join(log_dir, "evaluate.log"),
    mode="a",
    encoding="utf-8"
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(
    os.path.join(error_dir, "evaluation_errors.log"),
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

def load_test_data():
    x_path = os.path.join(data_dir, "X_test.csv")
    y_path = os.path.join(data_dir, "y_test.csv")
    try:
        X_test = pd.read_csv(x_path)
        y_test = pd.read_csv(y_path).squeeze("columns")
        logger.info(f"Test data yuklandi! X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        logger.error(f"Test fayllar topilmadi: {x_path} yoki {y_path}")
        return None, None
    except Exception as e:
        logger.error(f"Test data yuklashda xatolik: {e}")
        return None, None

def load_model():
    model_path = os.path.join(model_dir, "best_algorithm.pkl")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model yuklandi: {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model topilmadi: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Model yuklashda xatolik: {e}")
        return None

def evaluate_model():
    try:
        model = load_model()
        if model is None:
            return

        X_test, y_test = load_test_data()
        if X_test is None or y_test is None:
            return

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logger.info(f"R2: {r2:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MSE: {mse:.4f}")

        metrics_df = pd.DataFrame([{
            "R2": r2,
            "MAE": mae,
            "MSE": mse
        }])
        metrics_df.to_csv(os.path.join(result_dir, "evaluation_metrics.csv"), index=False)
        logger.info("evaluation_metrics.csv saqlandi")

        plt.figure(figsize=(6, 4))
        plt.axis("off")
        text = f"R2: {r2:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}"
        plt.text(0.05, 0.6, text, fontsize=14)
        plt.savefig(os.path.join(result_dir, "evaluation_result.png"), bbox_inches="tight")
        plt.close()
        logger.info("evaluation_result.png saqlandi")

        print(f"R2: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")

    except Exception as e:
        logger.error(f"Evaluate funksiyada xatolik: {e}")

if __name__ == "__main__":
    evaluate_model()







# from pathlib import Path
# import logging
# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# BASE_DIR = Path(__file__).resolve().parent.parent
# model_path = BASE_DIR / "models" / "best_algorithm.pkl"
# data_path = BASE_DIR / "Data" / "Preprocessed_Data" / "clean_data.csv"
# result_dir = BASE_DIR / "results"
# log_dir = BASE_DIR / "log"
# log_file = log_dir / "evaluate.log"

# log_dir.mkdir(exist_ok=True)
# result_dir.mkdir(exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file, encoding="utf-8"),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)

# def evaluate_model():
#     try:
#         logger.info("Evaluation started.")
#         logger.info(f"Model path: {model_path}")
#         logger.info(f"Data path: {data_path}")

#         if not model_path.exists():
#             logger.error(f"Model file not found: {model_path}")
#             return

#         if not data_path.exists():
#             logger.error(f"Data file not found: {data_path}")
#             return

#         model = joblib.load(model_path)
#         logger.info("Model loaded successfully.")

#         df = pd.read_csv(data_path)
#         logger.info(f"Data loaded successfully. Shape: {df.shape}")

#         if "Close" not in df.columns:
#             logger.error("Target column 'Close' not found in dataset.")
#             return

#         X = df.drop(columns=["Close"])
#         y = df["Close"]

#         logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")

#         y_pred = model.predict(X)
#         logger.info("Prediction completed.")

#         r2 = r2_score(y, y_pred)
#         mae = mean_absolute_error(y, y_pred)
#         mse = mean_squared_error(y, y_pred)

#         metrics_df = pd.DataFrame([{
#             "R2": r2,
#             "MAE": mae,
#             "MSE": mse
#         }])
#         metrics_path = result_dir / "evaluation_metrics.csv"
#         metrics_df.to_csv(metrics_path, index=False)
#         logger.info(f"Metrics saved to {metrics_path}")

#         plt.figure(figsize=(6, 4))
#         plt.axis("off")
#         text = f"R2: {r2:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}"
#         plt.text(0.05, 0.6, text, fontsize=14)
#         image_path = result_dir / "evaluation_result.png"
#         plt.savefig(image_path, bbox_inches="tight")
#         plt.close()
#         logger.info(f"Result image saved to {image_path}")

#         logger.info(f"R2: {r2:.4f}")
#         logger.info(f"MAE: {mae:.4f}")
#         logger.info(f"MSE: {mse:.4f}")

#     except Exception as e:
#         logger.exception(f"Evaluation failed: {e}")

# if __name__ == "__main__":
#     evaluate_model()

