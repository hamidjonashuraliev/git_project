import os
import pandas as pd

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR,
                             "Data",
                               "Raw_Data",
                               "Global football events and player performance.csv")

    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.info())