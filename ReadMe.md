<!-- fullWidth: false tocVisible: false tableWrap: true -->
<div align="center">



# 🚀 Git Project

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,100:8b5cf6&height=180&section=header&text=Machine%20Learning%20Pipeline&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35" />

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Holat-Active-success?style=for-the-badge&logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Project-orange?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

<p>
  <img src="https://skillicons.dev/icons?i=python,pandas,sklearn,xgboost,git,github" />
</p>

</div>

---

## ✨ Loyihaga umumiy qarash

Bu loyiha mashinani o‘qitish uchun to‘liq pipeline hisoblanadi.\
Unda data preprocessing, model training, evaluation va prediction bosqichlari tartibli tarzda joylashtirilgan.

---

## 🎯 Loyihaning maqsadi

Bu loyiha quyidagi vazifalarni bajaradi:

- original yoki tayyorlangan datani yuklash.
- datani tozalash va tayyorlash.
- bir nechta modelni train qilish.
- eng yaxshi modelni tanlash.
- test set ustida baholash.
- yangi data uchun prediction olish.
- log va error fayllarini saqlash.

---

## 🧱 Papkalar tuzilmasi

```bash
git_project/
│
├── Data/
│   ├── raw_data.csv
│   └── Preprocessed_Data/
│       ├── clean_data.csv
│       ├── X_test.csv
│       └── y_test.csv
│
├── models/
│   └── best_algorithm.pkl
│
├── results/
│   ├── train_metrics.csv
│   ├── evaluation_metrics.csv
│   └── evaluation_result.png
│
├── log/
│   ├── train.log
│   ├── evaluate.log
│   ├── predict.log
│   └── preprocess.log
│
├── errors/
│   ├── training_errors.log
│   ├── evaluation_errors.log
│   ├── testing_errors.log
│   └── preprocess_errors.log
│
├── pipeline/
│   ├── train_pipeline.py
│   ├── predict_pipeline.py
│   ├── preprocess.py
│   └── ReadMe.md
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── README.md
```

---

## ⚙️ Asosiy imkoniyatlar

- 🧹 Data cleaning va preprocessing.
- 🤖 Bir nechta modelni train qilish.
- 🏆 Eng yaxshi modelni tanlash.
- 📊 Test set ustida evaluation.
- 🔮 Yangi data ustida prediction.
- 📝 Log va error yuritish.
- 💾 Model va natijalarni saqlash.

---

## 🛠️ Ishlatilgan texnologiyalar

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-1E1E1E?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

---

## 📦 Fayllar vazifasi

### 🔹 `pipeline/preprocess.py`

Raw data bo‘lsa, uni tozalaydi va `clean_data.csv` yaratadi.

### 🔹 `scripts/train.py`

Modelni train qiladi, eng yaxshisini saqlaydi va `X_test.csv`, `y_test.csv` ni yozadi.

### 🔹 `scripts/evaluate.py`

Saqlangan modelni test setda baholaydi.

### 🔹 `scripts/predict.py`

Yangi data uchun prediction qiladi.

### 🔹 `pipeline/train_pipeline.py`

Train workflow’ni boshqaradi.

### 🔹 `pipeline/predict_pipeline.py`

Prediction workflow’ni boshqaradi.

---

## ▶️ Qanday ishga tushirish

### 1\. Data tayyorlash

```bash
python pipeline/preprocess.py
```

### 2\. Model train qilish

```bash
python pipeline/train_pipeline.py
```

### 3\. Modelni baholash

```bash
python scripts/evaluate.py
```

### 4\. Prediction olish

```bash
python scripts/predict.py
```

---

## 📊 Natijalar

- `models/best_algorithm.pkl` — saqlangan eng yaxshi model.
- `results/train_metrics.csv` — training metrikalari.
- `results/evaluation_metrics.csv` — final test metrikalari.
- `results/evaluation_result.png` — visual natija.
- `log/` — barcha loglar.
- `errors/` — barcha error loglar.

---

## 🧩 Muhim eslatmalar

- `clean_data.csv` — training uchun asosiy dataset.
- `X_test.csv` va `y_test.csv` — evaluation uchun saqlanadi.
- Barcha path’lar loyiha root’iga nisbatan yozilgan.
- Loyiha debug qilish va kengaytirish uchun qulay strukturada tuzilgan.

---

## 👨‍💻 Muallif

**Hamidjon**\
Machine Learning & Data Science Project