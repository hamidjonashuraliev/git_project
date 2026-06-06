import joblib
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Modelni yuklash ───────────────────────────────────
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
model = joblib.load(model_path)

print("=" * 50)
print(f"Model turi:       {type(model).__name__}")
print(f"Model parametrlari: {model.get_params()}")

# ─── Feature importance ────────────────────────────────
if hasattr(model, "feature_importances_"):
    data_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
    df = pd.read_csv(data_path)

    features = df.drop(columns=["Close"]).columns
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("\n" + "=" * 50)
    print("TOP 10 muhim ustunlar:")
    print("=" * 50)
    print(importance_df.head(10).to_string(index=False))

print("=" * 50)
print("Model muvaffaqiyatli yuklandi!")