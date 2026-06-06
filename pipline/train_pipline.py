import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from scripts.train import load_data, split_data, train_models, save_test_data, save_best_model

def run_train_pipeline():
    df = load_data()
    if df is None:
        print("Data yuklanmadi.")
        return

    X_train, X_test, y_train, y_test = split_data(df)
    save_test_data(X_test, y_test)

    results = train_models(X_train, X_test, y_train, y_test)
    save_best_model(results)

    print("Train pipeline tugadi.")

if __name__ == "__main__":
    run_train_pipeline()