# import os
# import sys
# import logging
# import pandas as pd
# import joblib

# # UTF-8 encoding
# sys.stdout.reconfigure(encoding='utf-8')

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR, "scripts"))

# # ─── Logging ───────────────────────────────────────────
# log_dir = os.path.join(BASE_DIR, "log")
# os.makedirs(log_dir, exist_ok=True)

# logging.basicConfig(
#     filename=os.path.join(log_dir, "test.log"),
#     filemode='a',
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)


# # ─── 1. Data yuklash testi ─────────────────────────────
# def test_data_load():
#     print("\n" + "=" * 50)
#     print("1. DATA LOAD TESTI")
#     print("=" * 50)
#     try:
#         from data_load import load_data
#         df = load_data()
#         assert df is not None,               "Data None!"
#         assert len(df) > 0,                  "Data bosh!"
#         assert "Close" in df.columns,        "Close ustuni yoq!"
#         assert "Ticker" in df.columns,       "Ticker ustuni yoq!"
#         assert df.isnull().sum().sum() == 0, "Bosh qiymatlar bor!"
#         print(f"[OK] Data yuklandi! Shape: {df.shape}")
#         print(f"[OK] Kompaniyalar: {list(df['Ticker'].unique())}")
#         logger.info("Data load testi muvaffaqiyatli!")
#         return True
#     except Exception as e:
#         print(f"[FAIL] Xatolik: {e}")
#         logger.error(f"Data load testi xato: {e}")
#         return False


# # ─── 2. Preprocessing testi ────────────────────────────
# def test_preprocessing():
#     print("\n" + "=" * 50)
#     print("2. PREPROCESSING TESTI")
#     print("=" * 50)
#     try:
#         clean_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#         assert os.path.exists(clean_path), "clean_data.csv topilmadi!"

#         df = pd.read_csv(clean_path)
#         assert df.isnull().sum().sum() == 0, "Bosh qiymatlar bor!"
#         assert "Close" in df.columns,        "Close ustuni yoq!"
#         print(f"[OK] Preprocessing! Shape: {df.shape}")
#         print(f"[OK] Bosh qiymatlar: {df.isnull().sum().sum()}")
#         logger.info("Preprocessing testi muvaffaqiyatli!")
#         return True
#     except Exception as e:
#         print(f"[FAIL] Xatolik: {e}")
#         logger.error(f"Preprocessing testi xato: {e}")
#         return False


# # ─── 3. Model testi ────────────────────────────────────
# def test_model():
#     print("\n" + "=" * 50)
#     print("3. MODEL TESTI")
#     print("=" * 50)
#     try:
#         model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
#         assert os.path.exists(model_path), "best_model.pkl topilmadi!"

#         model = joblib.load(model_path)
#         print(f"[OK] Model yuklandi: {type(model).__name__}")

#         clean_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#         df = pd.read_csv(clean_path)
#         X = df.drop(columns=["Close"])
#         y = df["Close"]

#         y_pred = model.predict(X[:10])
#         assert len(y_pred) == 10, "Bashorat xato!"
#         print(f"[OK] Bashorat ishlaydi!")
#         print(f"[OK] Namuna bashorat: {y_pred[:3].round(2)}")
#         logger.info("Model testi muvaffaqiyatli!")
#         return True
#     except Exception as e:
#         print(f"[FAIL] Xatolik: {e}")
#         logger.error(f"Model testi xato: {e}")
#         return False


# # ─── Umumiy natija ─────────────────────────────────────
# if __name__ == "__main__":
#     logger.info("Testing boshlandi")
#     print("\nTESTING BOSHLANDI...")

#     results = {
#         "Data Load":     test_data_load(),
#         "Preprocessing": test_preprocessing(),
#         "Model":         test_model(),
#     }

#     print("\n" + "=" * 50)
#     print("UMUMIY NATIJA:")
#     print("=" * 50)
#     for test, passed in results.items():
#         status = "[PASSED]" if passed else "[FAILED]"
#         print(f"{test:20} | {status}")

#     passed = sum(results.values())
#     total  = len(results)
#     print(f"\n{passed}/{total} test muvaffaqiyatli!")
#     logger.info(f"Testing tugadi: {passed}/{total} passed")












# import os
# import sys
# import logging
# import pandas as pd
# import joblib

# sys.stdout.reconfigure(encoding='utf-8')

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR, "scripts"))

# log_dir = os.path.join(BASE_DIR, "log")
# os.makedirs(log_dir, exist_ok=True)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.handlers.clear()
# logger.propagate = False

# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# test_handler = logging.FileHandler(
#     os.path.join(log_dir, "test.log"),
#     mode='a',
#     encoding='utf-8'
# )
# test_handler.setLevel(logging.INFO)
# test_handler.setFormatter(formatter)

# logger.addHandler(test_handler)


# def test_data_load():
#     print("\n" + "=" * 50)
#     print("1. DATA LOAD TESTI")
#     print("=" * 50)
#     try:
#         from data_load import load_data
#         df = load_data()
#         assert df is not None, "Data None!"
#         assert len(df) > 0, "Data bosh!"
#         assert "Close" in df.columns, "Close ustuni yoq!"
#         assert "Ticker" in df.columns, "Ticker ustuni yoq!"
#         assert df.isnull().sum().sum() == 0, "Bosh qiymatlar bor!"
#         print(f"[OK] Data yuklandi! Shape: {df.shape}")
#         print(f"[OK] Kompaniyalar: {list(df['Ticker'].unique())}")
#         logger.info("Data load testi muvaffaqiyatli!")
#         return True
#     except Exception as e:
#         print(f"[FAIL] Xatolik: {e}")
#         logger.error(f"Data load testi xato: {e}")
#         return False


# def test_preprocessing():
#     print("\n" + "=" * 50)
#     print("2. PREPROCESSING TESTI")
#     print("=" * 50)
#     try:
#         clean_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#         assert os.path.exists(clean_path), "clean_data.csv topilmadi!"

#         df = pd.read_csv(clean_path)
#         assert df.isnull().sum().sum() == 0, "Bosh qiymatlar bor!"
#         assert "Close" in df.columns, "Close ustuni yoq!"
#         print(f"[OK] Preprocessing! Shape: {df.shape}")
#         print(f"[OK] Bosh qiymatlar: {df.isnull().sum().sum()}")
#         logger.info("Preprocessing testi muvaffaqiyatli!")
#         return True
#     except Exception as e:
#         print(f"[FAIL] Xatolik: {e}")
#         logger.error(f"Preprocessing testi xato: {e}")
#         return False


# def test_model():
#     print("\n" + "=" * 50)
#     print("3. MODEL TESTI")
#     print("=" * 50)
#     try:
#         model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
#         assert os.path.exists(model_path), "best_model.pkl topilmadi!"

#         model = joblib.load(model_path)
#         print(f"[OK] Model yuklandi: {type(model).__name__}")

#         clean_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
#         df = pd.read_csv(clean_path)
#         X = df.drop(columns=["Close"])

#         y_pred = model.predict(X[:10])
#         assert len(y_pred) == 10, "Bashorat xato!"
#         print(f"[OK] Bashorat ishlaydi!")
#         print(f"[OK] Namuna bashorat: {y_pred[:3].round(2)}")
#         logger.info("Model testi muvaffaqiyatli!")
#         return True
#     except Exception as e:
#         print(f"[FAIL] Xatolik: {e}")
#         logger.error(f"Model testi xato: {e}")
#         return False


# if __name__ == "__main__":
#     logger.info("Testing boshlandi")
#     print("\nTESTING BOSHLANDI...")

#     results = {
#         "Data Load": test_data_load(),
#         "Preprocessing": test_preprocessing(),
#         "Model": test_model(),
#     }

#     print("\n" + "=" * 50)
#     print("UMUMIY NATIJA:")
#     print("=" * 50)
#     for test, passed in results.items():
#         status = "[PASSED]" if passed else "[FAILED]"
#         print(f"{test:20} | {status}")

#     passed = sum(results.values())
#     total = len(results)
#     print(f"\n{passed}/{total} test muvaffaqiyatli!")
#     logger.info(f"Testing tugadi: {passed}/{total} passed")


#  logger.error("TEST ERROR")






import os
import sys
import logging
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "scripts"))

log_dir = os.path.join(BASE_DIR, "log")
error_dir = os.path.join(BASE_DIR, "errors")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

logger = logging.getLogger("testing_logger")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

info_handler = logging.FileHandler(
    os.path.join(log_dir, "test.log"),
    mode="a",
    encoding="utf-8"
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(
    os.path.join(error_dir, "testing_errors.log"),
    mode="a",
    encoding="utf-8"
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)


def test_data_load():
    print("\n" + "=" * 50)
    print("1. DATA LOAD TESTI")
    print("=" * 50)
    try:
        from data_load import load_data
        df = load_data()
        assert df is not None, "Data None!"
        assert len(df) > 0, "Data bosh!"
        assert "Close" in df.columns, "Close ustuni yoq!"
        assert "Ticker" in df.columns, "Ticker ustuni yoq!"
        assert df.isnull().sum().sum() == 0, "Bosh qiymatlar bor!"
        print(f"[OK] Data yuklandi! Shape: {df.shape}")
        print(f"[OK] Kompaniyalar: {list(df['Ticker'].unique())}")
        logger.info("Data load testi muvaffaqiyatli!")
        return True
    except Exception as e:
        print(f"[FAIL] Xatolik: {e}")
        logger.error(f"Data load testi xato: {e}")
        return False


def test_preprocessing():
    print("\n" + "=" * 50)
    print("2. PREPROCESSING TESTI")
    print("=" * 50)
    try:
        clean_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
        assert os.path.exists(clean_path), "clean_data.csv topilmadi!"

        df = pd.read_csv(clean_path)
        assert df.isnull().sum().sum() == 0, "Bosh qiymatlar bor!"
        assert "Close" in df.columns, "Close ustuni yoq!"
        print(f"[OK] Preprocessing! Shape: {df.shape}")
        print(f"[OK] Bosh qiymatlar: {df.isnull().sum().sum()}")
        logger.info("Preprocessing testi muvaffaqiyatli!")
        return True
    except Exception as e:
        print(f"[FAIL] Xatolik: {e}")
        logger.error(f"Preprocessing testi xato: {e}")
        return False


def test_model():
    print("\n" + "=" * 50)
    print("3. MODEL TESTI")
    print("=" * 50)
    try:
        model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
        assert os.path.exists(model_path), "best_model.pkl topilmadi!"

        model = joblib.load(model_path)
        print(f"[OK] Model yuklandi: {type(model).__name__}")

        clean_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
        df = pd.read_csv(clean_path)
        X = df.drop(columns=["Close"])

        y_pred = model.predict(X[:10])
        assert len(y_pred) == 10, "Bashorat xato!"
        print(f"[OK] Bashorat ishlaydi!")
        print(f"[OK] Namuna bashorat: {y_pred[:3].round(2)}")
        logger.info("Model testi muvaffaqiyatli!")
        return True
    except Exception as e:
        print(f"[FAIL] Xatolik: {e}")
        logger.error(f"Model testi xato: {e}")
        return False


if __name__ == "__main__":
    logger.info("Testing boshlandi")
    print("\nTESTING BOSHLANDI...")

    results = {
        "Data Load": test_data_load(),
        "Preprocessing": test_preprocessing(),
        "Model": test_model(),
    }

    print("\n" + "=" * 50)
    print("UMUMIY NATIJA:")
    print("=" * 50)
    for test, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{test:20} | {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n{passed}/{total} test muvaffaqiyatli!")
    logger.info(f"Testing tugadi: {passed}/{total} passed")

    logger.error("TEST ERROR")