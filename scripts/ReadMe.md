<!-- fullWidth: false tocVisible: false tableWrap: true -->
# Stock Price Prediction System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)\
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)\
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)\
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)

> Aksiya narxlarini bashorat qiluvchi Machine Learning tizimi

---

## Loyiha haqida

Bu loyiha Yahoo Finance dan real aksiya narxlarini yuklab,\
ularni qayta ishlab, ML modellar yordamida narxni bashorat qiladi.\
12 ta yirik kompaniyaning 1 yillik narx tarixi ishlatiladi.

---

## Loyiha tuzilmasi

aksiya_loyiha/\
├── scripts/\
│   ├── data_load.py          # Aksiya narxlarini yuklash\
│   ├── preprocessing.py      # Ma'lumotlarni tozalash\
│   ├── train.py              # Modellarni o'rgatish\
│   ├── model_info.py         # Model ma'lumotlari\
│   └── test.py               # Testlar\
├── Data/\
│   ├── Raw_Data/             # Yuklab olingan narxlar\
│   └── Preprocessed_Data/    # Tozalangan ma'lumotlar\
├── models/\
│   └── best_model.pkl        # Eng yaxshi model\
├── log/                      # Log fayllar\
├── Notebooks/                # Jupyter notebooks\
└── ReadMe.md

---

## Kompaniyalar

| Kompaniya | Ticker | Soha            |
| --------- | ------ | --------------- |
| Apple     | AAPL   | Texnologiya     |
| Tesla     | TSLA   | Elektr mashina  |
| Microsoft | MSFT   | Texnologiya     |
| Google    | GOOGL  | Texnologiya     |
| Amazon    | AMZN   | E-commerce      |
| Meta      | META   | Ijtimoiy tarmoq |
| Nvidia    | NVDA   | Chip / AI       |
| Netflix   | NFLX   | Streaming       |
| Intel     | INTC   | Chip            |
| AMD       | AMD    | Chip            |
| TSMC      | TSM    | Chip            |
| Alibaba   | BABA   | E-commerce      |

---

## Dataset

| Ustun    | Ma'nosi                     |
| -------- | --------------------------- |
| Date     | Sana                        |
| Open     | Kun boshidagi narx          |
| High     | Eng yuqori narx             |
| Low      | Eng past narx               |
| Close    | Kun oxiridagi narx (TARGET) |
| Volume   | Savdo hajmi                 |
| Ticker   | Kompaniya kodi              |
| Currency | Valyuta (USD)               |

---

## O'rnatish

### 1\. Loyihani klonlash

```bash
git clone https://github.com/username/aksiya_loyiha.git
cd aksiya_loyiha
```

### 2\. Virtual muhit yaratish

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3\. Kutubxonalarni o'rnatish

```bash
pip install -r requirements.txt
```

**requirements.txt:**\
pandas\
numpy\
scikit-learn\
xgboost\
yfinance\
joblib\
matplotlib\
seaborn

---

## Ishga tushirish

```bash
# 1. Data yuklash
python scripts/data_load.py

# 2. Preprocessing
python scripts/preprocessing.py

# 3. Model o'rgatish
python scripts/train.py

# 4. Model ma'lumotlari
python scripts/model_info.py

# 5. Test
python scripts/test.py
```

---

## Modellar

| Model            | Vazifasi              |
| ---------------- | --------------------- |
| LinearRegression | Asosiy chiziqli model |
| RandomForest     | Daraxtlar ansambli    |
| XGBoost          | Gradient boosting     |

Eng yaxshi model R2 score asosida tanlanadi va\
`models/best_model.pkl` ga saqlanadi.

---




