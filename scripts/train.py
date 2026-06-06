import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
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
os.makedirs(model_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

info_handler = logging.FileHandler(
    os.path.join(log_dir, "train.log"),
    mode="a",
    encoding="utf-8"
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(
    os.path.join(error_dir, "training_errors.log"),
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

def load_data():
    file_path = os.path.join(data_dir, "clean_data.csv")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data yuklandi! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Fayl topilmadi: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Xatolik: {e}")
        return None

def split_data(df):
    X = df.drop(columns=["Close"])
    y = df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        try:
            logger.info(f"{name} o'qitilmoqda...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results[name] = {"model": model, "r2": r2, "mae": mae}
            logger.info(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")
            print(f"{name:20} | R2: {r2:.4f} | MAE: {mae:.4f}")

        except Exception as e:
            logger.error(f"{name} xatolik: {e}")

    return results

def save_best_model(results):
    if not results:
        logger.error("Hech qanday model train bo'lmadi.")
        return None

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = results[best_name]["model"]
    best_r2 = results[best_name]["r2"]
    best_mae = results[best_name]["mae"]

    model_path = os.path.join(model_dir, "best_algorithm.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Eng yaxshi model: {best_name}")
    logger.info(f"Model saqlandi: {model_path}")

    metrics_df = pd.DataFrame([{
        "best_model": best_name,
        "R2": best_r2,
        "MAE": best_mae
    }])
    metrics_df.to_csv(os.path.join(result_dir, "train_metrics.csv"), index=False)

    return best_name, model_path

def save_test_data(X_test, y_test):
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    logger.info(f"Test data saqlandi: {X_test_path}")
    logger.info(f"Test target saqlandi: {y_test_path}")

def main():
    try:
        df = load_data()
        if df is None:
            return

        X_train, X_test, y_train, y_test = split_data(df)
        save_test_data(X_test, y_test)

        results = train_models(X_train, X_test, y_train, y_test)
        save_best_model(results)

        logger.info("Training tugadi.")
    except Exception as e:
        logger.error(f"Main xatolik: {e}")

if __name__ == "__main__":
    main()






# import os
# import logging
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# from xgboost import XGBRegressor
# import joblib
# import matplotlib.pyplot as plt

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# log_dir = os.path.join(BASE_DIR, "log")
# error_dir = os.path.join(BASE_DIR, "errors")
# os.makedirs(log_dir, exist_ok=True)
# os.makedirs(error_dir, exist_ok=True)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.handlers.clear()
# logger.propagate = False

# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# info_handler = logging.FileHandler(
#     os.path.join(log_dir, "train.log"),
#     mode="a",
#     encoding="utf-8"
# )
# info_handler.setLevel(logging.INFO)
# info_handler.setFormatter(formatter)

# error_handler = logging.FileHandler(
#     os.path.join(error_dir, "training_errors.log"),
#     mode="a",
#     encoding="utf-8"
# )
# error_handler.setLevel(logging.ERROR)
# error_handler.setFormatter(formatter)

# logger.addHandler(info_handler)
# logger.addHandler(error_handler)


# def load_data():
#     file_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#     try:
#         df = pd.read_csv(file_path)
#         logger.info(f"Data yuklandi! Shape: {df.shape}")
#         return df
#     except FileNotFoundError:
#         logger.error(f"Fayl topilmadi: {file_path}")
#         return None
#     except Exception as e:
#         logger.error(f"Xatolik: {e}")
#         return None


# def split_data(df):
#     X = df.drop(columns=["Close"])
#     y = df["Close"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
#     return X_train, X_test, y_train, y_test


# def train_models(X_train, X_test, y_train, y_test):
#     models = {
#         "LinearRegression": LinearRegression(),
#         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
#     }

#     results = {}

#     for name, model in models.items():
#         try:
#             logger.info(f"{name} o'qitilmoqda...")
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             r2 = r2_score(y_test, y_pred)
#             mae = mean_absolute_error(y_test, y_pred)

#             results[name] = {"model": model, "r2": r2, "mae": mae}
#             logger.info(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")
#             logger.info(f"{name} to'g'ri ishladi")
#             print(f"{name:20} | R2: {r2:.4f} | MAE: {mae:.4f}")

#         except Exception as e:
#             logger.error(f"{name} xatolik: {e}")

#     return results






# def save_best_model(results):
#     if not results:
#         logger.error("Hech bir model muvaffaqiyatli o'qitilmadi.")
#         return

#     best_name = max(results, key=lambda x: results[x]["r2"])
#     best_model = results[best_name]["model"]

#     logger.info(f"Eng yaxshi model: {best_name} — R2: {results[best_name]['r2']:.4f}")
#     print(f"\nEng yaxshi model: {best_name}")
#     print(f"R2: {results[best_name]['r2']:.4f}")
#     print(f"MAE: {results[best_name]['mae']:.4f}")

#     save_dir = os.path.join(BASE_DIR, "models")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "best_model.pkl")

#     joblib.dump(best_model, save_path)
#     logger.info(f"Model saqlandi: {save_path}")
#     print(f"Model saqlandi: {save_path}")

#     model = joblib.load(save_path)
#     print(type(model))


# def save_best_algorithm(results):
#     if not results:
#         logger.error("Hech bir algorithm muvaffaqiyatli o'qitilmadi.")
#         return

#     best_name = max(results, key=lambda x: results[x]["r2"])
#     best_algorithm = results[best_name]["model"]

#     best_r2 = results[best_name]["r2"]
#     best_mae = results[best_name]["mae"]

#     logger.info(f"Eng yaxshi algorithm: {best_name} — R2: {best_r2:.4f}")
#     print(f"\nEng yaxshi algorithm: {best_name}")
#     print(f"R2: {best_r2:.4f}")
#     print(f"MAE: {best_mae:.4f}")

#     save_dir = os.path.join(BASE_DIR, "models")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "best_algorithm.pkl")

#     joblib.dump(best_algorithm, save_path)
#     logger.info(f"Algorithm saqlandi: {save_path}")
#     print(f"Algorithm saqlandi: {save_path}")

#     plt.figure(figsize=(6, 4))
#     plt.axis("off")
#     text = (
#         f"Best algorithm: {best_name}\n"
#         f"R2: {best_r2:.4f}\n"
#         f"MAE: {best_mae:.4f}"
#     )
#     plt.text(0.05, 0.6, text, fontsize=14)
#     plt.savefig(os.path.join(save_dir, "best_algorithm_result.png"), bbox_inches="tight")
#     plt.close()

#     plt.figure(figsize=(6, 4))
#     plt.axis("off")
#     text2 = f"Algorithm saved successfully:\n{save_path}"
#     plt.text(0.05, 0.6, text2, fontsize=14)
#     plt.savefig(os.path.join(save_dir, "algorithm_result.png"), bbox_inches="tight")
#     plt.close()

#     plt.figure(figsize=(6, 4))
#     plt.axis("off")
#     text2 = f"algorithm saved successfully:\n{save_path}"
#     plt.text(0.05, 0.6, text2, fontsize=14)
#     plt.savefig(os.path.join(save_dir, "algorithm_result.png"), bbox_inches="tight")
#     plt.close()

# if __name__ == "__main__":
#     logger.info("Train skript boshlandi")

#     df = load_data()
#     if df is not None:
#         X_train, X_test, y_train, y_test = split_data(df)
#         results = train_models(X_train, X_test, y_train, y_test)
#         save_best_algorithm(results)

#     logger.info("Train skript tugadi")




# import os
# import logging
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# from xgboost import XGBRegressor
# import joblib

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log_dir = os.path.join(BASE_DIR, "log")
# os.makedirs(log_dir, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(log_dir, "train.log"),
#     filemode='a',
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# def load_data():
#     file_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#     try:
#         df = pd.read_csv(file_path)
#         logger.info(f"Data yuklandi! Shape: {df.shape}")
#         return df
#     except FileNotFoundError:
#         logger.error(f"Fayl topilmadi: {file_path}")
#         return None
#     except Exception as e:
#         logger.error(f"Xatolik: {e}")
#         return None

# def split_data(df):
#     X = df.drop(columns=["Close"])
#     y = df["Close"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
#     return X_train, X_test, y_train, y_test

# def train_models(X_train, X_test, y_train, y_test):
#     models = {
#         "LinearRegression": LinearRegression(),
#         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
#     }

#     results = {}

#     for name, model in models.items():
#         try:
#             logger.info(f"{name} o'qitilmoqda...")
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             r2 = r2_score(y_test, y_pred)
#             mae = mean_absolute_error(y_test, y_pred)

#             results[name] = {"model": model, "r2": r2, "mae": mae}
#             logger.info(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")
#             print(f"{name:20} | R2: {r2:.4f} | MAE: {mae:.4f}")

#         except Exception as e:
#             logger.error(f"{name} xatolik: {e}")

#     return results

# def save_best_model(results):
#     if not results:
#         logger.error("Hech bir model muvaffaqiyatli o'qitilmadi.")
#         return

#     best_name = max(results, key=lambda x: results[x]["r2"])
#     best_model = results[best_name]["model"]

#     logger.info(f"Eng yaxshi model: {best_name} — R2: {results[best_name]['r2']:.4f}")
#     print(f"\nEng yaxshi model: {best_name}")
#     print(f"R2: {results[best_name]['r2']:.4f}")
#     print(f"MAE: {results[best_name]['mae']:.4f}")

#     save_dir = os.path.join(BASE_DIR, "models")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "best_model.pkl")

#     joblib.dump(best_model, save_path)
#     logger.info(f"Model saqlandi: {save_path}")
#     print(f"Model saqlandi: {save_path}")

# if __name__ == "__main__":
#     logger.info("Train skript boshlandi")

#     df = load_data()
#     if df is not None:
#         X_train, X_test, y_train, y_test = split_data(df)
#         results = train_models(X_train, X_test, y_train, y_test)
#         save_best_model(results)

#     logger.info("Train skript tugadi")








# import os
# import logging
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# from xgboost import XGBRegressor
# import joblib

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# log_dir = os.path.join(BASE_DIR, "log")
# error_dir = os.path.join(BASE_DIR, "errors")
# os.makedirs(log_dir, exist_ok=True)
# os.makedirs(error_dir, exist_ok=True)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.handlers.clear()

# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# train_handler = logging.FileHandler(
#     os.path.join(log_dir, "train.log"),
#     mode='a',
#     encoding='utf-8'
# )
# train_handler.setLevel(logging.INFO)
# train_handler.setFormatter(formatter)

# error_handler = logging.FileHandler(
#     os.path.join(error_dir, "training_errors.log"),
#     mode='a',
#     encoding='utf-8'
# )
# error_handler.setLevel(logging.ERROR)
# error_handler.setFormatter(formatter)

# logger.addHandler(train_handler)
# logger.addHandler(error_handler)


# def load_data():
#     file_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#     try:
#         df = pd.read_csv(file_path)
#         logger.info(f"Data yuklandi! Shape: {df.shape}")
#         return df
#     except FileNotFoundError:
#         logger.error(f"Fayl topilmadi: {file_path}")
#         return None
#     except Exception as e:
#         logger.error(f"Xatolik: {e}")
#         return None


# def split_data(df):
#     X = df.drop(columns=["Close"])
#     y = df["Close"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
#     return X_train, X_test, y_train, y_test


# def train_models(X_train, X_test, y_train, y_test):
#     models = {
#         "LinearRegression": LinearRegression(),
#         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
#     }

#     results = {}

#     for name, model in models.items():
#         try:
#             logger.info(f"{name} o'qitilmoqda...")
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             r2 = r2_score(y_test, y_pred)
#             mae = mean_absolute_error(y_test, y_pred)

#             results[name] = {"model": model, "r2": r2, "mae": mae}
#             logger.info(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")
#             print(f"{name:20} | R2: {r2:.4f} | MAE: {mae:.4f}")

#         except Exception as e:
#             logger.error(f"{name} xatolik: {e}")

#     return results


# def save_best_model(results):
#     if not results:
#         logger.error("Hech bir model muvaffaqiyatli o'qitilmadi.")
#         return

#     best_name = max(results, key=lambda x: results[x]["r2"])
#     best_model = results[best_name]["model"]

#     logger.info(f"Eng yaxshi model: {best_name} — R2: {results[best_name]['r2']:.4f}")
#     print(f"\nEng yaxshi model: {best_name}")
#     print(f"R2: {results[best_name]['r2']:.4f}")
#     print(f"MAE: {results[best_name]['mae']:.4f}")

#     save_dir = os.path.join(BASE_DIR, "models")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "best_model.pkl")

#     joblib.dump(best_model, save_path)
#     logger.info(f"Model saqlandi: {save_path}")
#     print(f"Model saqlandi: {save_path}")


# if __name__ == "__main__":
#     logger.info("Train skript boshlandi")

#     df = load_data()
#     if df is not None:
#         X_train, X_test, y_train, y_test = split_data(df)
#         results = train_models(X_train, X_test, y_train, y_test)
#         save_best_model(results)

#     logger.info("Train skript tugadi")









# import os
# import logging
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# from xgboost import XGBRegressor
# import joblib

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# log_dir = os.path.join(BASE_DIR, "log")
# error_dir = os.path.join(BASE_DIR, "errors")
# os.makedirs(log_dir, exist_ok=True)
# os.makedirs(error_dir, exist_ok=True)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.handlers.clear()
# logger.propagate = False

# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# info_handler = logging.FileHandler(
#     os.path.join(log_dir, "train.log"),
#     mode="a",
#     encoding="utf-8"
# )
# info_handler.setLevel(logging.INFO)
# info_handler.setFormatter(formatter)

# error_handler = logging.FileHandler(
#     os.path.join(error_dir, "training_errors.log"),
#     mode="a",
#     encoding="utf-8"
# )
# error_handler.setLevel(logging.ERROR)
# error_handler.setFormatter(formatter)

# logger.addHandler(info_handler)
# logger.addHandler(error_handler)


# def load_data():
#     file_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#     try:
#         df = pd.read_csv(file_path)
#         logger.info(f"Data yuklandi! Shape: {df.shape}")
#         return df
#     except FileNotFoundError:
#         logger.error(f"Fayl topilmadi: {file_path}")
#         return None
#     except Exception as e:
#         logger.error(f"Xatolik: {e}")
#         return None


# def split_data(df):
#     X = df.drop(columns=["Close"])
#     y = df["Close"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
#     return X_train, X_test, y_train, y_test


# def train_models(X_train, X_test, y_train, y_test):
#     models = {
#         "LinearRegression": LinearRegression(),
#         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
#     }

#     results = {}

#     for name, model in models.items():
#         try:
#             logger.info(f"{name} o'qitilmoqda...")
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             r2 = r2_score(y_test, y_pred)
#             mae = mean_absolute_error(y_test, y_pred)

#             results[name] = {"model": model, "r2": r2, "mae": mae}
#             logger.info(f"{name} → R2: {r2:.4f}, MAE: {mae:.4f}")
#             print(f"{name:20} | R2: {r2:.4f} | MAE: {mae:.4f}")

#         except Exception as e:
#             logger.error(f"{name} xatolik: {e}")

#     return results


# def save_best_model(results):
#     if not results:
#         logger.error("Hech bir model muvaffaqiyatli o'qitilmadi.")
#         return

#     best_name = max(results, key=lambda x: results[x]["r2"])
#     best_model = results[best_name]["model"]

#     logger.info(f"Eng yaxshi model: {best_name} — R2: {results[best_name]['r2']:.4f}")
#     print(f"\nEng yaxshi model: {best_name}")
#     print(f"R2: {results[best_name]['r2']:.4f}")
#     print(f"MAE: {results[best_name]['mae']:.4f}")

#     save_dir = os.path.join(BASE_DIR, "models")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "best_model.pkl")

#     joblib.dump(best_model, save_path)
#     logger.info(f"Model saqlandi: {save_path}")
#     print(f"Model saqlandi: {save_path}")


# if __name__ == "__main__":
#     logger.info("Train skript boshlandi")

#     df = load_data()
#     if df is not None:
#         X_train, X_test, y_train, y_test = split_data(df)
#         results = train_models(X_train, X_test, y_train, y_test)
#         save_best_model(results)

#     logger.info("Train skript tugadi")








