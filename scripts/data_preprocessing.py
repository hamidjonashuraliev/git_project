import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 🔥 1. TOZALA
def tozala(df):
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df


# 🔥 2. ENCODING
def encodla(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
    return df


# 🔥 3. SCALING
def scale(df):
    scaler = MinMaxScaler()

    num_col=df.select_dtypes(include=['int64','float64']).columns.drop('FTR')
    df[num_col]=scaler.fit_transform(df[num_col])
    return df



# 🔥 MAIN PIPELINE
def preprocess():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    file_path = os.path.join(
        BASE_DIR,
        "Data",
        "Raw_Data",
        "Global football events and player performance.csv"
    )

    df = pd.read_csv(file_path)

    # 1. clean
    df = tozala(df)

    # 2. encode
    df = encodla(df)

    # 3. scale
    df = scale(df)

    # save
    save_path = os.path.join(BASE_DIR, "Data", "Preprocessed_Data", "clean_data.csv")
    df.to_csv(save_path, index=False)

    print("Preprocessing done ✅")


if __name__ == "__main__":
    preprocess()