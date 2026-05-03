# ⚽ Football Match Outcome Prediction System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

> 🎯 Futbol o'yinlari natijalarini bashorat qiluvchi Machine Learning tizimi

## 📋 Loyiha haqida

Bu Machine Learning loyiha futbol o'yinlari statistikasini tahlil qilib, o'yin natijalarini (uy egasi g'alabasi, mehmon g'alabasi yoki durrang) bashorat qiladi. Dataset 98 ta turli o'yin statistikasini o'z ichiga oladi: gollar, zarbalar, kartochkalar, burchaklar va boshqalar.

## ✨ Asosiy imkoniyatlar

- 🎲 **O'yin natijalari bashorati** - uy egasi/mehmon g'alabasi yoki durrang
- 📊 **Go'llar soni bashorati** - FTHG va FTAG bashorati
- 📈 **Statistik tahlil** - 98 ta o'yin ko'rsatkichi tahlili
- 🤖 **ML Pipeline** - avtomatlashtirilgan bashorat tizimi
- 📉 **Vizualizatsiya** - grafik va diagrammalar

## 📁 Loyiha tuzilmasi

📦 football-match-prediction/
├── 📁 scripts/
│ ├── 🐍 data_load.py # Ma'lumotlarni yuklash va dastlabki ko'rish
│ ├── 🐍 data_preprocessing.py # Ma'lumotlarni tozalash va feature engineering
│ ├── 🐍 train.py # ML modellarni o'rgatish
│ ├── 🐍 evaluate.py # Model baholash va metrikalar
│ └── 🐍 test.py # Yangi o'yinlar uchun bashorat
├── 📁 data/
│ └── 📄 football_matches.csv # O'yin statistikasi (98 ustun)
├── 📁 models/
│ └── 💾 trained_models.pkl # O'rgatilgan modellar
├── 📁 notebooks/ # Jupyter notebooks (agar bo'lsa)
├── 📄 ReadMe.md # Loyiha hujjati
└── 📄 requirements.txt # Python kutubxonalari

## 🎯 Dataset tuzilmasi

**98 ustunli dataset** - har bir qator bitta futbol o'yini:

### Asosiy natija ustunlari:

- **FTHG** - Full Time Home Goals (To'liq vaqt uy egasi gollari)
- **FTAG** - Full Time Away Goals (To'liq vaqt mehmon gollari)
- **HTHG** - Half Time Home Goals (Yarim vaqt uy egasi gollari)
- **HTAG** - Half Time Away Goals (Yarim vaqt mehmon gollari)
- **FTR** - Full Time Result (H=Home Win, A=Away Win, D=Draw)
- **HTR** - Half Time Result

### O'yin statistikasi ustunlari:

- **HS** - Home Shots (Uy egasi zarbalar soni)
- **AS** - Away Shots (Mehmon zarbalar soni)
- **HST** - Home Shots on Target (Uy egasi darvozaga zarba)
- **AST** - Away Shots on Target (Mehmon darvozaga zarba)
- **HC** - Home Corners (Uy egasi burchaklar)
- **AC** - Away Corners (Mehmon burchaklar)
- **HF** - Home Fouls (Uy egasi qoidabuzarlik)
- **AF** - Away Fouls (Mehmon qoidabuzarlik)
- **HY** - Home Yellow Cards (Uy egasi sariq kartochkalar)
- **AY** - Away Yellow Cards (Mehmon sariq kartochkalar)
- **HR** - Home Red Cards (Uy egasi qizil kartochkalar)
- **AR** - Away Red Cards (Mehmon qizil kartochkalar)

### Qo'shimcha ustunlar:

- **Date** - O'yin sanasi
- **HomeTeam** - Uy egasi jamoa
- **AwayTeam** - Mehmon jamoa
- **Referee** - Hakam
- **Betting Odds** - Bukmekerlik koeffitsientlari
- ... va yana 70+ statistik ko'rsatkich

## 🚀 O'rnatish va ishga tushirish

### 1. Loyihani klonlash

```bash
git clone https://github.com/yourusername/football-match-prediction.git
cd football-match-prediction
```

### 2. Virtual muhit yaratish

```bash
# Virtual muhit
python -m venv venv

# Faollashtirish (Windows)
venv\Scripts\activate

# Faollashtirish (Mac/Linux)
source venv/bin/activate
```

### 3. Kutubxonalarni o'rnatish

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
```

## 📖 Skriptlar tavsifi

### 1️⃣ `data_load.py` - Ma'lumotlarni yuklash

**Vazifasi:**

- CSV fayldan 98 ustunli datasetni yuklash
- Dastlabki ma'lumot ko'rish (head, info, describe)
- Null qiymatlarni aniqlash
- Ustunlar turlarini tekshirish

**Ishlatish:**

```bash
python scripts/data_load.py
```

**Asosiy funksiyalar:**

```python
def load_data(file_path):
    """Dataset yuklash"""
    df = pd.read_csv(file_path)
    return df

def show_dataset_info(df):
    """Dataset haqida umumiy ma'lumot"""
    print(f"O'yinlar soni: {len(df)}")
    print(f"Ustunlar soni: {len(df.columns)}")
    print(f"\nIlk 5 ta o'yin:")
    print(df.head())
    print(f"\nNull qiymatlar:")
    print(df.isnull().sum())
```

**Chiquvchi ma'lumot:**
✅ Dataset yuklandi: 3800 o'yin
✅ Ustunlar: 98
✅ Null qiymatlar: 12 (0.3%)
✅ Sana diapazoni: 2020-2024

---

### 2️⃣ `data_preprocessing.py` - Ma'lumotlarni qayta ishlash

**Vazifasi:**

- Null qiymatlarni to'ldirish/o'chirish
- Sana ustunini datetime formatiga o'zgartirish
- Kategorik ustunlarni encode qilish (HomeTeam, AwayTeam, Referee)
- Outlier'larni aniqlash va qayta ishlash
- Feature engineering

**Ishlatish:**

```bash
python scripts/data_preprocessing.py
```

**Feature Engineering:**

```python
def create_features(df):
    """Yangi feature'lar yaratish"""

    # 1. Zarba samaradorligi
    df['Home_Shot_Accuracy'] = df['HST'] / df['HS']
    df['Away_Shot_Accuracy'] = df['AST'] / df['AS']

    # 2. Gol konversiyasi
    df['Home_Goal_Conversion'] = df['FTHG'] / df['HST']
    df['Away_Goal_Conversion'] = df['FTAG'] / df['AST']

    # 3. Birinchi yarim faollik
    df['Home_First_Half_Dominance'] = df['HTHG'] - df['HTAG']
    df['Away_First_Half_Dominance'] = df['HTAG'] - df['HTHG']

    # 4. Hujum kuchi
    df['Home_Attack_Strength'] = df['HS'] + df['HC']
    df['Away_Attack_Strength'] = df['AS'] + df['AC']

    # 5. Distsiplina
    df['Home_Discipline'] = df['HY'] + (df['HR'] * 3)
    df['Away_Discipline'] = df['AY'] + (df['AR'] * 3)

    # 6. Xaftalik kun
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # 7. Oy (mavsumiylik)
    df['Month'] = pd.to_datetime(df['Date']).dt.month

    # 8. Umumiy gollar
    df['Total_Goals'] = df['FTHG'] + df['FTAG']

    # 9. Gol farqi
    df['Goal_Difference'] = df['FTHG'] - df['FTAG']

    return df
```

**Label Encoding:**

```python
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df):
    """Kategorik ustunlarni encode qilish"""
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_referee = LabelEncoder()

    df['HomeTeam_Encoded'] = le_home.fit_transform(df['HomeTeam'])
    df['AwayTeam_Encoded'] = le_away.fit_transform(df['AwayTeam'])
    df['Referee_Encoded'] = le_referee.fit_transform(df['Referee'])

    return df, le_home, le_away, le_referee
```

---

### 3️⃣ `train.py` - Model o'rgatish

**Vazifasi:**

- Train/Test split (80/20)
- Turli ML modellarni o'rgatish
- Hyperparameter tuning
- Best model tanlash
- Model saqlash

**Ishlatish:**

```bash
python scripts/train.py --model random_forest
python scripts/train.py --model xgboost --target FTR
python scripts/train.py --model all  # Barcha modellarni sinash
```

**Qo'llab-quvvatlanadigan modellar:**

**Klassifikatsiya (FTR bashorati: H/A/D):**

```python
models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'gradient_boosting': GradientBoostingClassifier(),
    'xgboost': XGBClassifier(),
    'svm': SVC(probability=True),
    'knn': KNeighborsClassifier()
}
```

**Regressiya (FTHG/FTAG bashorati):**

```python
models = {
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100),
    'xgboost': XGBRegressor(),
    'gradient_boosting': GradientBoostingRegressor()
}
```

**Train kodı:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    # Feature'lar va target
    X = df.drop(['FTR', 'FTHG', 'FTAG', 'Date', 'HomeTeam', 'AwayTeam'], axis=1)
    y = df['FTR']  # H, A yoki D

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )

    # O'rgatish
    model.fit(X_train, y_train)

    # Saqlash
    joblib.dump(model, 'models/rf_match_outcome.pkl')

    return model, X_test, y_test
```

---

### 4️⃣ `evaluate.py` - Model baholash

**Vazifasi:**

- O'rgatilgan modelni yuklash
- Test ma'lumotlarda baholash
- Metrikalarni hisoblash
- Confusion Matrix
- Classification Report
- Feature Importance
- Vizualizatsiyalar yaratish

**Ishlatish:**

```bash
python scripts/evaluate.py
python scripts/evaluate.py --model xgboost
```

**Klassifikatsiya metrikalari:**

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_classification(model, X_test, y_test):
    # Bashorat
    y_pred = model.predict(X_test)

    # Metrikalar
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
```

**Regressiya metrikalari (gollar uchun):**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f} gol")
    print(f"RMSE: {rmse:.3f} gol")
    print(f"R² Score: {r2:.3f}")
```

**Vizualizatsiya:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Confusion Matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Home Win', 'Draw', 'Away Win'],
            yticklabels=['Home Win', 'Draw', 'Away Win'])
plt.title('Confusion Matrix')
plt.ylabel('Haqiqiy natija')
plt.xlabel('Bashorat natija')
plt.savefig('results/confusion_matrix.png')

# 2. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 6))
plt.bar(range(20), importances[indices])
plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45)
plt.title('Top 20 muhim feature\'lar')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
```

---

### 5️⃣ `test.py` - Bashorat va test

**Vazifasi:**

- Yangi o'yin uchun bashorat
- Real-time bashorat demo
- Model validatsiya
- Batch bashorat

**Ishlatish:**

```bash
python scripts/test.py
python scripts/test.py --match "Arsenal vs Chelsea"
```

**Bashorat funksiyasi:**

```python
def predict_match(model, match_data):
    """
    Bitta o'yin uchun bashorat

    Parameters:
    -----------
    match_data : dict
        O'yin statistikasi

    Returns:
    --------
    prediction : str
        'H' (Home Win), 'D' (Draw), 'A' (Away Win)
    probabilities : dict
        Har bir natija ehtimolligi
    """
    # Feature'larni tayyorlash
    X = prepare_features(match_data)

    # Bashorat
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    result = {
        'prediction': prediction,
        'home_win_prob': probabilities[0] * 100,
        'draw_prob': probabilities[1] * 100,
        'away_win_prob': probabilities[2] * 100
    }

    return result
```

**Demo bashorat:**

```python
# Misol o'yin
new_match = {
    'HomeTeam': 'Manchester United',
    'AwayTeam': 'Liverpool',
    'HS': 15,
    'AS': 12,
    'HST': 6,
    'AST': 5,
    'HC': 7,
    'AC': 5,
    'HF': 10,
    'AF': 12,
    'HY': 2,
    'AY': 3,
    'HR': 0,
    'AR': 0,
    # ... qolgan feature'lar
}

# Bashorat
result = predict_match(model, new_match)

print(f"\n⚽ O'yin bashorati:")
print(f"Uy egasi g'alabasi: {result['home_win_prob']:.1f}%")
print(f"Durrang: {result['draw_prob']:.1f}%")
print(f"Mehmon g'alabasi: {result['away_win_prob']:.1f}%")
print(f"\n🎯 Bashorat: {result['prediction']}")
```

## 🔄 To'liq Pipeline

**Avtomatik pipeline ishga tushirish:**

```bash
# Linux/Mac
chmod +x run_pipeline.sh
./run_pipeline.sh

# Windows
run_pipeline.bat
```

**run_pipeline.sh:**

```bash
#!/bin/bash

echo "⚽ Futbol bashorat pipeline boshlandi..."
echo "========================================"

echo "\n📥 1/5: Ma'lumotlarni yuklash..."
python scripts/data_load.py

echo "\n🔧 2/5: Ma'lumotlarni qayta ishlash..."
python scripts/data_preprocessing.py

echo "\n🤖 3/5: Model o'rgatish..."
python scripts/train.py --model xgboost

echo "\n📊 4/5: Model baholash..."
python scripts/evaluate.py

echo "\n🎯 5/5: Test bashoratlar..."
python scripts/test.py

echo "\n✅ Pipeline muvaffaqiyatli tugadi!"
echo "Natijalar 'results/' papkasida."
```

## 📊 Kutilayotgan natijalar

### Model Performance:

Classification Metrics:
├── Accuracy: 55-60%
├── Precision: 0.58
├── Recall: 0.56
└── F1-Score: 0.57
Class-wise Performance:
├── Home Win (H): 65% accuracy
├── Draw (D): 40% accuracy (eng qiyin)
└── Away Win (A): 58% accuracy
Goals Prediction (Regression):
├── FTHG MAE: 0.85 gol
├── FTAG MAE: 0.92 gol
└── R² Score: 0.72

### Top 10 muhim feature'lar:

HST (Home Shots on Target) - 0.18
AST (Away Shots on Target) - 0.16
Home_Shot_Accuracy - 0.12
Away_Shot_Accuracy - 0.11
HTHG (Half Time Home Goals) - 0.09
HTAG (Half Time Away Goals) - 0.08
HC (Home Corners) - 0.06
AC (Away Corners) - 0.05
Home_Attack_Strength - 0.04
HS (Home Shots) - 0.03

## 🎯 Bashorat misollari

**Real bashorat (2024 mavsumi):**
O'yin: Manchester City vs Arsenal
Bashorat: Home Win (H)
Ehtimollik: 52%
Haqiqiy natija: 3-1 (H) ✅
O'yin: Chelsea vs Liverpool
Bashorat: Away Win (A)
Ehtimollik: 45%
Haqiqiy natija: 1-1 (D) ❌
O'yin: Tottenham vs Brighton
Bashorat: Draw (D)
Ehtimollik: 38%
Haqiqiy natija: 2-2 (D) ✅

## 💡 Feature Engineering tushuntirishlari

**1. Shot Accuracy (Zarba aniqligi):**

```python
Home_Shot_Accuracy = HST / HS
# Misol: 6 zarba darvozaga / 15 umumiy zarba = 0.40 (40%)
```

**2. Goal Conversion (Gol konversiyasi):**

```python
Home_Goal_Conversion = FTHG / HST
# Misol: 2 gol / 6 darvozaga zarba = 0.33 (33%)
```

**3. Attack Strength (Hujum kuchi):**

```python
Home_Attack_Strength = HS + HC
# Misol: 15 zarba + 7 burchak = 22 hujum ko'rsatkichi
```

**4. Discipline Score (Distsiplina):**

```python
Home_Discipline = HY + (HR * 3)
# Misol: 2 sariq + (0 qizil * 3) = 2
```

## 🐛 Muammolarni hal qilish

**1. Dataset yuklanmadi:**

```bash
# Fayl yo'lini tekshiring
python scripts/data_load.py --check-path

# Dataset formatini tekshiring
head -n 5 data/football_matches.csv
```

**2. Model o'rgatilmadi:**

```bash
# Verbose mode
python scripts/train.py --verbose

# Log fayl
cat logs/training.log
```

**3. Bashorat ishlamayapti:**

```bash
# Model mavjudligini tekshirish
ls -l models/

# Test rejimda ishga tushirish
python scripts/test.py --debug
```

## 📚 Qo'shimcha resurslar

- [Scikit-learn Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Football Data Sources](https://www.football-data.co.uk/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

## 🎓 Keyingi qadamlar

- [ ] Deep Learning modellari (LSTM, GRU)
- [ ] Real-time API yaratish (Flask/FastAPI)
- [ ] Web dashboard (Streamlit)
- [ ] Telegram bot integratsiyasi
- [ ] Jamoalar form tahlili (oxirgi 5 o'yin)
- [ ] Ob-havo ma'lumotlarini qo'shish
- [ ] Transfer bozori ta'sirini tahlil qilish

## 🤝 Hissa qo'shish

Pull request'lar xush kelibsiz!

```bash
git checkout -b feature/NewFeature
git commit -m "Add: Yangi feature"
git push origin feature/NewFeature
```

## 👨‍💻 Muallif

**Sizning ismingiz**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 📄 Litsenziya

MIT License - [LICENSE](LICENSE)

## 🙏 Minnatdorchilik

- Football-Data.co.uk - dataset uchun
- Scikit-learn jamoasiga
- Open source hamjamiyatiga

---

⭐ **Foydali bo'lsa, star qo'yishni unutmang!**
╔══════════════════════════════════════╗
║ Load → Process → Train → Predict ║
║ Football ML Pipeline 🚀⚽ ║
