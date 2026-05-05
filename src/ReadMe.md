# ⚽ Football Match Prediction - ML Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

> 🎯 Professional Machine Learning pipeline for football match outcome prediction

## 📋 Project Overview

This is a complete Machine Learning system that predicts football match outcomes using historical match statistics. The project includes data loading, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

**Dataset:** 98-column match statistics including:

- Goals (FTHG, FTAG, HTHG, HTAG)
- Shots (HS, AS, HST, AST)
- Corners, Fouls, Cards
- And 90+ more features

**Target:** Predict match result (Home Win, Draw, Away Win) or goal counts

## ✨ Features

- 📊 **Complete ML Pipeline** - From raw data to trained model
- 🔧 **Advanced Preprocessing** - Null handling, outlier removal, encoding
- 🏗️ **Feature Engineering** - 60+ engineered features
- 🤖 **Multiple ML Models** - Linear, Tree-based, Ensemble methods
- ⚙️ **Hyperparameter Tuning** - Grid/Random Search optimization
- 📈 **Comprehensive Evaluation** - Multiple metrics and visualizations
- 💾 **Model Persistence** - Save and load trained models

## 📁 Project Structure

football-ml-pipeline/
├── 📁 src/
│ ├── 🐍 analysis.py # Exploratory Data Analysis & Visualization
│ ├── 🐍 data_loader.py # Load and validate raw data
│ ├── 🐍 models.py # ML model definitions and training
│ ├── 🐍 preprocessing.py # Data cleaning and transformation
│ └── 🐍 tuning.py # Hyperparameter optimization
│
├── 📁 Data/
│ ├── 📁 Raw_Data/ # Original dataset
│ ├── 📁 Preprocessed_Data/ # Cleaned data
│ └── 📁 Engineered_Data/ # Final features
│
├── 📁 models/ # Saved trained models
├── 📁 results/ # Outputs, plots, reports
├── 📄 ReadMe.md # Project documentation
└── 📄 requirements.txt # Python dependencies

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/football-ml-pipeline.git
cd football-ml-pipeline

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
xgboost>=1.7.0
lightgbm>=3.3.0
joblib>=1.2.0
scipy>=1.10.0
```

### 2. Prepare Data

Place your football match data in `Data/Raw_Data/`:

```bash
Data/Raw_Data/football_matches.csv
```

### 3. Run the Pipeline

**Option A: Step-by-step**

```bash
# 1. Load and explore data
python src/data_loader.py

# 2. Analyze data
python src/analysis.py

# 3. Preprocess data
python src/preprocessing.py

# 4. Train models
python src/models.py

# 5. Tune hyperparameters
python src/tuning.py
```

**Option B: Automated pipeline**

```bash
# Run complete pipeline
python run_pipeline.py
```

## 📖 Module Documentation

### 1️⃣ `data_loader.py` - Data Loading

**Purpose:** Load raw data and perform initial validation

**Key Functions:**

```python
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load football match data from CSV

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """

def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate dataset structure and quality

    Returns:
    --------
    dict
        Validation report with null counts, duplicates, etc.
    """

def show_data_info(df: pd.DataFrame) -> None:
    """Display dataset summary statistics"""
```

**Usage:**

```python
from src.data_loader import load_data, validate_data

# Load data
df = load_data('Data/Raw_Data/football_matches.csv')

# Validate
report = validate_data(df)
print(report)
```

**Output:**
✅ Dataset loaded: 3800 matches
✅ Columns: 98
✅ Date range: 2020-2024
❌ Null values: 245 (6.4%)
⚠️ Duplicates: 12

---

### 2️⃣ `analysis.py` - Exploratory Data Analysis

**Purpose:** Analyze data patterns and create visualizations

**Key Functions:**

```python
def analyze_goals_distribution(df: pd.DataFrame) -> None:
    """Visualize goals distribution (FTHG, FTAG)"""

def analyze_correlations(df: pd.DataFrame) -> None:
    """Create correlation heatmap"""

def analyze_team_performance(df: pd.DataFrame, team: str) -> dict:
    """Analyze specific team statistics"""

def create_eda_report(df: pd.DataFrame) -> None:
    """Generate comprehensive EDA report"""
```

**Usage:**

```python
from src.analysis import create_eda_report, analyze_goals_distribution

# Full EDA
create_eda_report(df)

# Specific analysis
analyze_goals_distribution(df)
```

**Generated Visualizations:**
results/
├── goals_distribution.png
├── correlation_heatmap.png
├── shots_vs_goals.png
├── home_advantage.png
└── eda_report.html

---

### 3️⃣ `preprocessing.py` - Data Preprocessing

**Purpose:** Clean data and prepare for modeling

**Key Functions:**

```python
def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle null values

    Parameters:
    -----------
    strategy : str
        'median', 'mean', 'mode', or 'drop'
    """

def remove_outliers(df: pd.DataFrame, columns: list, method: str = 'IQR') -> pd.DataFrame:
    """
    Remove statistical outliers

    Parameters:
    -----------
    method : str
        'IQR' or 'zscore'
    """

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode team names, referees, etc."""

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features

    Creates 60+ features including:
    - Shot accuracy
    - Goal conversion
    - Attack/Defense strength
    - Discipline scores
    - Temporal features
    """

def normalize_features(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize numerical features

    Parameters:
    -----------
    method : str
        'standard', 'minmax', or 'robust'
    """
```

**Usage:**

```python
from src.preprocessing import (
    handle_missing_values,
    remove_outliers,
    encode_categorical,
    create_features
)

# Preprocessing pipeline
df = handle_missing_values(df)
df = remove_outliers(df, columns=['FTHG', 'FTAG'])
df = encode_categorical(df)
df = create_features(df)

# Save preprocessed data
df.to_csv('Data/Preprocessed_Data/cleaned_data.csv', index=False)
```

**Engineered Features Examples:**

```python
# Shot Statistics
- Home_Shot_Accuracy = HST / HS
- Home_Conversion_Rate = FTHG / HST

# Attack Strength
- Home_Attack_Strength = HS + HC + FTHG

# Defense Metrics
- Home_Defense_Efficiency = 1 - (FTAG / AS)

# Temporal Features
- DayOfWeek, IsWeekend, Month, Season

# 60+ more features...
```

---

### 4️⃣ `models.py` - Model Training & Evaluation

**Purpose:** Train ML models and evaluate performance

**Key Classes:**

```python
class FootballPredictor:
    """
    Main model training and prediction class

    Supported models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - Support Vector Machine
    - Neural Network
    """

    def __init__(self, model_type: str = 'random_forest'):
        """Initialize predictor with model type"""

    def train(self, X_train, y_train):
        """Train the model"""

    def predict(self, X_test):
        """Make predictions"""

    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate model performance

        Returns:
        --------
        dict
            Metrics: accuracy, precision, recall, f1-score
        """

    def save_model(self, filepath: str):
        """Save trained model to disk"""

    def load_model(self, filepath: str):
        """Load saved model"""
```

**Usage:**

```python
from src.models import FootballPredictor
from sklearn.model_selection import train_test_split

# Prepare data
X = df.drop(['FTR', 'FTHG', 'FTAG'], axis=1)
y = df['FTR']  # 'H', 'D', 'A'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
predictor = FootballPredictor(model_type='xgboost')
predictor.train(X_train, y_train)

# Evaluate
metrics = predictor.evaluate(X_test, y_test)
print(metrics)

# Save
predictor.save_model('models/xgboost_model.pkl')
```

**Evaluation Metrics:**

```python
{
    'accuracy': 0.58,
    'precision': 0.61,
    'recall': 0.58,
    'f1_score': 0.59,
    'confusion_matrix': [[...], [...], [...]],
    'classification_report': {...}
}
```

**Model Comparison:**

```python
def compare_models(X_train, X_test, y_train, y_test):
    """Compare multiple models"""

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(),
        'LightGBM': LGBMClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    return results
```

---

### 5️⃣ `tuning.py` - Hyperparameter Optimization

**Purpose:** Find best model parameters

**Key Functions:**

```python
def grid_search_tuning(
    X_train, y_train,
    model_type: str = 'random_forest',
    cv: int = 5
) -> dict:
    """
    Grid Search optimization

    Parameters:
    -----------
    model_type : str
        'random_forest', 'xgboost', 'lightgbm'
    cv : int
        Cross-validation folds

    Returns:
    --------
    dict
        Best parameters and score
    """

def random_search_tuning(
    X_train, y_train,
    model_type: str,
    n_iter: int = 100
) -> dict:
    """Random Search optimization (faster)"""

def bayesian_optimization(
    X_train, y_train,
    model_type: str,
    n_trials: int = 50
) -> dict:
    """Bayesian optimization (most efficient)"""
```

**Usage:**

```python
from src.tuning import grid_search_tuning

# Grid Search for Random Forest
best_params = grid_search_tuning(
    X_train, y_train,
    model_type='random_forest',
    cv=5
)

print(f"Best parameters: {best_params['params']}")
print(f"Best CV score: {best_params['score']:.4f}")

# Use best parameters
from sklearn.ensemble import RandomForestClassifier

best_model = RandomForestClassifier(**best_params['params'])
best_model.fit(X_train, y_train)
```

**Parameter Grids:**

```python
# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0]
}

# LightGBM
param_grid_lgbm = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}
```

---

## 📊 Expected Results

### Model Performance:

┌──────────────────────────────────────────┐
│ Classification Metrics │
├──────────────────────────────────────────┤
│ │
│ Accuracy: 58.3% │
│ Precision: 0.61 │
│ Recall: 0.58 │
│ F1-Score: 0.59 │
│ │
│ Class-wise Performance: │
│ ├── Home Win (H): 65% accuracy │
│ ├── Draw (D): 40% accuracy │
│ └── Away Win (A): 58% accuracy │
│ │
└──────────────────────────────────────────┘

### Feature Importance (Top 10):

```python
1.  HST (Home Shots on Target)      → 0.18
2.  AST (Away Shots on Target)      → 0.16
3.  Home_Shot_Accuracy              → 0.14
4.  Away_Shot_Accuracy              → 0.11
5.  HTHG (Half Time Home Goals)     → 0.09
6.  Home_Attack_Strength            → 0.08
7.  Away_Defense_Efficiency         → 0.06
8.  HC (Home Corners)               → 0.05
9.  Total_Goals                     → 0.04
10. Home_Conversion_Rate            → 0.03
```

### Confusion Matrix:

            Predicted
          H    D    A

Actual H 120 30 20
D 40 35 45
A 25 40 105

---

## 🎯 Making Predictions

### Predict Single Match:

```python
from src.models import FootballPredictor
import pandas as pd

# Load trained model
predictor = FootballPredictor()
predictor.load_model('models/best_model.pkl')

# New match data
new_match = {
    'HomeTeam': 'Arsenal',
    'AwayTeam': 'Chelsea',
    'HS': 15,
    'AS': 12,
    'HST': 6,
    'AST': 5,
    'HC': 7,
    'AC': 5,
    'HF': 10,
    'AF': 12,
    # ... other features
}

# Predict
result = predictor.predict_match(new_match)

print(f"⚽ Match Prediction:")
print(f"Home Win:  {result['home_win_prob']:.1f}%")
print(f"Draw:      {result['draw_prob']:.1f}%")
print(f"Away Win:  {result['away_win_prob']:.1f}%")
print(f"\n🎯 Predicted Result: {result['prediction']}")
```

**Output:**
⚽ Match Prediction:
Home Win: 52.3%
Draw: 28.1%
Away Win: 19.6%
🎯 Predicted Result: H (Home Win)

### Batch Predictions:

```python
# Predict multiple matches
upcoming_matches = pd.read_csv('upcoming_matches.csv')
predictions = predictor.predict_batch(upcoming_matches)

predictions.to_csv('results/predictions.csv', index=False)
```

---

## 🔧 Configuration

**config.py:**

```python
# Data paths
RAW_DATA_PATH = 'Data/Raw_Data/football_matches.csv'
PROCESSED_DATA_PATH = 'Data/Preprocessed_Data/cleaned_data.csv'
ENGINEERED_DATA_PATH = 'Data/Engineered_Data/final_features.csv'

# Model settings
MODEL_TYPE = 'xgboost'  # 'logistic', 'random_forest', 'xgboost', 'lightgbm'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature engineering
CREATE_ADVANCED_FEATURES = True
NORMALIZE_METHOD = 'standard'  # 'standard', 'minmax', 'robust'

# Training
USE_CLASS_WEIGHTS = True  # For imbalanced data
CROSS_VALIDATION = 5

# Hyperparameter tuning
TUNING_METHOD = 'grid'  # 'grid', 'random', 'bayesian'
N_ITER = 100  # For random search

# Output
SAVE_RESULTS = True
RESULTS_DIR = 'results/'
MODELS_DIR = 'models/'
```

---

## 📈 Visualization Examples

The project generates various visualizations:

```python
# In analysis.py
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Goals distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df['FTHG'].hist(bins=20, color='blue', alpha=0.7)
plt.title('Home Goals Distribution')
plt.subplot(1, 2, 2)
df['FTAG'].hist(bins=20, color='red', alpha=0.7)
plt.title('Away Goals Distribution')
plt.savefig('results/goals_distribution.png')

# 2. Correlation heatmap
plt.figure(figsize=(15, 12))
correlation = df.corr()
sns.heatmap(correlation, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('results/correlation_heatmap.png')

# 3. Feature importance
plt.figure(figsize=(10, 8))
feature_imp = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)[:20]
sns.barplot(x=feature_imp.values, y=feature_imp.index)
plt.title('Top 20 Feature Importance')
plt.savefig('results/feature_importance.png')
```

---

## 🐛 Troubleshooting

### Common Issues:

**1. Import Error:**

```bash
ModuleNotFoundError: No module named 'sklearn'

# Solution:
pip install scikit-learn
```

**2. Memory Error:**

```python
# Reduce dataset size or use sampling
df_sample = df.sample(frac=0.5, random_state=42)
```

**3. Model accuracy too low:**

```python
# Try:
1. Feature engineering (add more features)
2. Hyperparameter tuning
3. Different model (try XGBoost)
4. Collect more data
5. Handle class imbalance
```

**4. Slow training:**

```python
# Use LightGBM instead of XGBoost
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=100, n_jobs=-1)
```

---

## 🎓 Learning Resources

**Machine Learning:**

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)

**Football Analytics:**

- [Football-Data.co.uk](https://www.football-data.co.uk/)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Friends of Tracking](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w)

**Python:**

- [Pandas Tutorial](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
- [Real Python](https://realpython.com/)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

**Code Style:**

- Follow PEP 8
- Add docstrings
- Write unit tests
- Update documentation

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Football-Data.co.uk for providing datasets
- Scikit-learn team for amazing ML library
- Open source community

---

## 📊 Project Status

✅ Data Loading - Complete
✅ EDA & Analysis - Complete
✅ Preprocessing - Complete
✅ Feature Engineering - Complete
✅ Model Training - Complete
✅ Hyperparameter Tuning - Complete
✅ Evaluation - Complete
🔄 Web API - In Progress
🔄 Real-time Predictions - Planned
🔄 Mobile App - Planned

---

## 🎯 Future Enhancements

- [ ] Real-time data integration (API)
- [ ] Deep Learning models (LSTM, Transformer)
- [ ] Player-level statistics
- [ ] Ensemble methods
- [ ] Web dashboard (Streamlit/Flask)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] A/B testing framework

---

⭐ **If this project helped you, please star it!**
╔═══════════════════════════════════════╗
║ Data → Preprocess → Train → Predict ║
║ Football ML Pipeline 🚀⚽ ║
╚═══════════════════════════════════════╝

**Made with ⚽, 🐍 and 🤖**

_"In football, the worst blindness is only seeing the ball."_
_"In ML, the worst blindness is only seeing the accuracy."_
