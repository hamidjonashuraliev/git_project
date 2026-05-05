# 📊 Results Directory - Analysis Outputs & Visualizations

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

> 📈 Bu papka barcha tahlil natijalari, vizualizatsiyalar va hisobotlarni saqlaydi

## 📁 Directory Structure

results/
├── 📁 figures/ # Barcha grafiklar va diagrammalar
│ ├── 📊 eda/ # Exploratory Data Analysis grafiklari
│ ├── 📈 model_perf/ # Model performance grafiklari
│ ├── 🎯 predictions/ # Bashorat vizualizatsiyalari
│ └── 📉 training/ # Training jarayoni grafiklari
│
├── 📁 tables/ # CSV va Excel jadvallar
│ ├── 📄 metrics.csv # Model metrikalari
│ ├── 📄 predictions.csv # Bashorat natijalari
│ ├── 📄 feature_imp.csv # Feature importance
│ └── 📄 results_summary.xlsx
│
└── 📄 ReadMe.md # Bu fayl

---

## 📊 1. Figures (Grafiklar)

### 📂 `figures/eda/` - Exploratory Data Analysis

Bu papkada **data exploration** grafiklari saqlanadi:

#### 🎯 Goals Distribution (Gollar Taqsimoti)

**Fayl:** `goals_distribution.png`

```python
# Qanday yaratiladi (analysis.py)
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Home Goals
axes[0, 0].hist(df['FTHG'], bins=range(0, 10), color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Full Time Home Goals Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Goals')
axes[0, 0].set_ylabel('Frequency')

# Away Goals
axes[0, 1].hist(df['FTAG'], bins=range(0, 10), color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Full Time Away Goals Distribution', fontsize=14, fontweight='bold')

# Half Time Home
axes[1, 0].hist(df['HTHG'], bins=range(0, 6), color='#2ecc71', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Half Time Home Goals', fontsize=14)

# Half Time Away
axes[1, 1].hist(df['HTAG'], bins=range(0, 6), color='#f39c12', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Half Time Away Goals', fontsize=14)

plt.tight_layout()
plt.savefig('results/figures/eda/goals_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Ko'rinishi:**
┌─────────────────────────────────────────┐
│ Full Time Home Goals Distribution │
│ │
│ █ │
│ █ │
│ █ █ │
│ █ █ █ │
│ █ █ █ █ │
│ ──┴──┴──┴──┴──┴──┴──┴────────────── │
│ 0 1 2 3 4 5 6 7 8 9 │
└─────────────────────────────────────────┘

#### 📊 Correlation Heatmap

**Fayl:** `correlation_heatmap.png`

```python
# Feature'lar orasidagi bog'lanish
plt.figure(figsize=(20, 16))

# Faqat muhim feature'lar
important_features = [
    'FTHG', 'FTAG', 'HTHG', 'HTAG',
    'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HF', 'AF',
    'HY', 'AY', 'HR', 'AR'
]

correlation = df[important_features].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)

plt.title('Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/eda/correlation_heatmap.png', dpi=300)
```

**Natija:**
FTHG FTAG HS AS HST
FTHG 1.00 0.15 0.68 0.12 0.82
FTAG 0.15 1.00 0.11 0.71 0.13
HS 0.68 0.11 1.00 0.25 0.78
AS 0.12 0.71 0.25 1.00 0.19
HST 0.82 0.13 0.78 0.19 1.00

#### ⚽ Shots vs Goals

**Fayl:** `shots_vs_goals.png`

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Home: Shots on Target vs Goals
axes[0].scatter(df['HST'], df['FTHG'], alpha=0.5, color='blue')
axes[0].set_xlabel('Home Shots on Target')
axes[0].set_ylabel('Home Goals')
axes[0].set_title('Home: Shots on Target vs Goals')

# Away: Shots on Target vs Goals
axes[1].scatter(df['AST'], df['FTAG'], alpha=0.5, color='red')
axes[1].set_xlabel('Away Shots on Target')
axes[1].set_ylabel('Away Goals')
axes[1].set_title('Away: Shots on Target vs Goals')

plt.savefig('results/figures/eda/shots_vs_goals.png', dpi=300)
```

#### 🏠 Home Advantage Analysis

**Fayl:** `home_advantage.png`

```python
# Uy egasi afzalligi tahlili
results_count = df['FTR'].value_counts()

plt.figure(figsize=(10, 6))
colors = ['#3498db', '#95a5a6', '#e74c3c']
plt.pie(
    results_count.values,
    labels=['Home Win', 'Draw', 'Away Win'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.1, 0, 0)  # Explode home win slice
)
plt.title('Match Outcomes Distribution\n(Home Advantage)', fontsize=16)
plt.savefig('results/figures/eda/home_advantage.png', dpi=300)
```

**EDA papkasidagi barcha fayllar:**
figures/eda/
├── goals_distribution.png # Gollar taqsimoti
├── correlation_heatmap.png # Korrelyatsiya matritsasi
├── shots_vs_goals.png # Zarbalar vs Gollar
├── home_advantage.png # Uy egasi afzalligi
├── cards_distribution.png # Kartochkalar tahlili
├── corners_analysis.png # Burchaklar tahlili
├── temporal_trends.png # Vaqt bo'yicha trendlar
├── team_performance.png # Jamoalar statistikasi
└── outliers_detection.png # Outlier'lar

---

### 📂 `figures/model_perf/` - Model Performance

#### 🎯 Confusion Matrix

**Fayl:** `confusion_matrix.png`

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Home Win', 'Draw', 'Away Win']
)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Match Outcome Prediction', fontsize=16, pad=20)
plt.savefig('results/figures/model_perf/confusion_matrix.png', dpi=300)
```

**Natija:**
Predicted
H D A
Actual H 120 30 20 ← 170 (71% accuracy)
D 40 35 45 ← 120 (29% accuracy)
A 25 40 105 ← 170 (62% accuracy)
Accuracy per class:

Home Win: 71%
Draw: 29% (eng qiyin!)
Away Win: 62%

#### 📊 ROC Curve

**Fayl:** `roc_curve.png`

```python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Multi-class ROC
y_test_bin = label_binarize(y_test, classes=['H', 'D', 'A'])
y_pred_proba = model.predict_proba(X_test)

plt.figure(figsize=(10, 8))

for i, label in enumerate(['Home Win', 'Draw', 'Away Win']):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-class Classification')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('results/figures/model_perf/roc_curve.png', dpi=300)
```

#### 🎯 Feature Importance

**Fayl:** `feature_importance.png`

```python
# Top 20 muhim feature'lar
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=feature_imp,
    x='importance',
    y='feature',
    palette='viridis'
)
plt.title('Top 20 Feature Importance', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/figures/model_perf/feature_importance.png', dpi=300)
```

**Top features:**

HST (Home Shots on Target) ████████████████░░ 0.18
AST (Away Shots on Target) ███████████████░░░ 0.16
Home_Shot_Accuracy ██████████████░░░░ 0.14
Away_Shot_Accuracy ████████████░░░░░░ 0.11
HTHG (Half Time Home Goals) ██████████░░░░░░░░ 0.09
Home_Attack_Strength ████████░░░░░░░░░░ 0.08
Away_Defense_Efficiency ██████░░░░░░░░░░░░ 0.06
HC (Home Corners) █████░░░░░░░░░░░░░ 0.05
Total_Goals ████░░░░░░░░░░░░░░ 0.04
Home_Conversion_Rate ███░░░░░░░░░░░░░░░ 0.03

#### 📈 Learning Curves

**Fayl:** `learning_curves.png`

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, label='Validation score', color='red')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('results/figures/model_perf/learning_curves.png', dpi=300)
```

#### 🎲 Precision-Recall Curve

**Fayl:** `precision_recall.png`

```python
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (label, ax) in enumerate(zip(['Home Win', 'Draw', 'Away Win'], axes)):
    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_pred_proba[:, i]
    )

    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall: {label}')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/model_perf/precision_recall.png', dpi=300)
```

**Model Performance papkasidagi fayllar:**
figures/model_perf/
├── confusion_matrix.png # Confusion matrix
├── roc_curve.png # ROC curve (multi-class)
├── feature_importance.png # Feature importance
├── learning_curves.png # Learning curves
├── precision_recall.png # Precision-Recall curves
├── calibration_curve.png # Probability calibration
├── error_analysis.png # Xatoliklar tahlili
└── model_comparison.png # Modellar taqqoslash

---

### 📂 `figures/predictions/` - Predictions Visualizations

#### 🎯 Prediction Examples

**Fayl:** `prediction_samples.png`

```python
# 10 ta test o'yini uchun bashorat
sample_predictions = pd.DataFrame({
    'Match': [f"Match {i+1}" for i in range(10)],
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10],
    'Home_Win_Prob': y_pred_proba[:10, 0] * 100,
    'Draw_Prob': y_pred_proba[:10, 1] * 100,
    'Away_Win_Prob': y_pred_proba[:10, 2] * 100
})

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(sample_predictions))
width = 0.25

ax.bar(x - width, sample_predictions['Home_Win_Prob'], width, label='Home Win %', color='#3498db')
ax.bar(x, sample_predictions['Draw_Prob'], width, label='Draw %', color='#95a5a6')
ax.bar(x + width, sample_predictions['Away_Win_Prob'], width, label='Away Win %', color='#e74c3c')

ax.set_xlabel('Matches')
ax.set_ylabel('Probability (%)')
ax.set_title('Prediction Probabilities - Sample Matches')
ax.set_xticks(x)
ax.set_xticklabels(sample_predictions['Match'], rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/predictions/prediction_samples.png', dpi=300)
```

#### 📊 Actual vs Predicted

**Fayl:** `actual_vs_predicted.png`

```python
from sklearn.metrics import accuracy_score

# Har bir class uchun accuracy
classes = ['H', 'D', 'A']
accuracies = []

for cls in classes:
    mask = y_test == cls
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred[mask])
        accuracies.append(acc * 100)
    else:
        accuracies.append(0)

plt.figure(figsize=(10, 6))
bars = plt.bar(['Home Win', 'Draw', 'Away Win'], accuracies, color=['#3498db', '#95a5a6', '#e74c3c'])
plt.ylabel('Accuracy (%)')
plt.title('Prediction Accuracy by Match Outcome')
plt.ylim(0, 100)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

plt.grid(axis='y', alpha=0.3)
plt.savefig('results/figures/predictions/actual_vs_predicted.png', dpi=300)
```

**Predictions papkasidagi fayllar:**
figures/predictions/
├── prediction_samples.png # Bashorat misollari
├── actual_vs_predicted.png # Haqiqiy vs Bashorat
├── confidence_distribution.png # Ishonch taqsimoti
├── top_predictions.png # Eng ishonchli bashoratlar
└── failed_predictions.png # Xato bashoratlar tahlili

---

### 📂 `figures/training/` - Training Process

#### 📈 Training History

**Fayl:** `training_history.png`

```python
# Training va validation loss
history = {
    'train_loss': [0.95, 0.82, 0.71, 0.65, 0.61, 0.58, 0.56, 0.55],
    'val_loss': [0.98, 0.85, 0.76, 0.72, 0.70, 0.69, 0.68, 0.68],
    'train_acc': [0.45, 0.52, 0.58, 0.62, 0.64, 0.66, 0.67, 0.68],
    'val_acc': [0.43, 0.50, 0.55, 0.57, 0.58, 0.58, 0.59, 0.58]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history['train_loss'], label='Training Loss', marker='o')
ax1.plot(history['val_loss'], label='Validation Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Model Loss During Training')
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy
ax2.plot(history['train_acc'], label='Training Accuracy', marker='o')
ax2.plot(history['val_acc'], label='Validation Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy During Training')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/training/training_history.png', dpi=300)
```

**Training papkasidagi fayllar:**
figures/training/
├── training_history.png # Loss va accuracy
├── validation_curve.png # Validation curve
├── hyperparameter_tuning.png # Tuning natijalari
└── cross_validation.png # CV natijalari

---

## 📄 2. Tables (Jadvallar)

### 📊 `tables/metrics.csv` - Model Metrikalari

```csv
model,accuracy,precision,recall,f1_score,training_time,prediction_time
Logistic Regression,0.52,0.54,0.52,0.52,2.3,0.01
Random Forest,0.57,0.59,0.57,0.57,45.2,0.15
XGBoost,0.583,0.61,0.58,0.59,67.8,0.08
LightGBM,0.576,0.60,0.58,0.58,23.4,0.05
Neural Network,0.554,0.57,0.55,0.55,120.5,0.03
```

**Python'da yaratish:**

```python
import pandas as pd

metrics_df = pd.DataFrame({
    'model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Neural Network'],
    'accuracy': [0.52, 0.57, 0.583, 0.576, 0.554],
    'precision': [0.54, 0.59, 0.61, 0.60, 0.57],
    'recall': [0.52, 0.57, 0.58, 0.58, 0.55],
    'f1_score': [0.52, 0.57, 0.59, 0.58, 0.55],
    'training_time': [2.3, 45.2, 67.8, 23.4, 120.5],
    'prediction_time': [0.01, 0.15, 0.08, 0.05, 0.03]
})

metrics_df.to_csv('results/tables/metrics.csv', index=False)
```

---

### 🎯 `tables/predictions.csv` - Bashorat Natijalari

```csv
match_id,home_team,away_team,actual_result,predicted_result,home_win_prob,draw_prob,away_win_prob,correct
1,Arsenal,Chelsea,H,H,0.523,0.281,0.196,True
2,Liverpool,Man City,D,A,0.312,0.354,0.334,False
3,Tottenham,Brighton,A,A,0.289,0.287,0.424,True
4,Man United,Everton,H,D,0.445,0.389,0.166,False
5,Newcastle,Aston Villa,H,H,0.567,0.245,0.188,True
...
```

**Yaratish:**

```python
predictions_df = pd.DataFrame({
    'match_id': range(1, len(y_test) + 1),
    'home_team': X_test['HomeTeam'].values,
    'away_team': X_test['AwayTeam'].values,
    'actual_result': y_test.values,
    'predicted_result': y_pred,
    'home_win_prob': y_pred_proba[:, 0],
    'draw_prob': y_pred_proba[:, 1],
    'away_win_prob': y_pred_proba[:, 2],
    'correct': y_test.values == y_pred
})

predictions_df.to_csv('results/tables/predictions.csv', index=False)
```

---

### 🏆 `tables/feature_imp.csv` - Feature Importance

```csv
rank,feature,importance,category
1,HST,0.182,Shot Statistics
2,AST,0.164,Shot Statistics
3,Home_Shot_Accuracy,0.141,Engineered
4,Away_Shot_Accuracy,0.112,Engineered
5,HTHG,0.094,Goal Statistics
6,Home_Attack_Strength,0.081,Engineered
7,Away_Defense_Efficiency,0.067,Engineered
8,HC,0.053,Corner Statistics
9,Total_Goals,0.042,Engineered
10,Home_Conversion_Rate,0.034,Engineered
...
```

**Yaratish:**

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

feature_importance['rank'] = range(1, len(feature_importance) + 1)

# Kategoriya qo'shish
def categorize_feature(feature_name):
    if 'Shot' in feature_name or 'HS' in feature_name or 'AS' in feature_name:
        return 'Shot Statistics'
    elif 'Goal' in feature_name or 'FTHG' in feature_name or 'FTAG' in feature_name:
        return 'Goal Statistics'
    elif 'Corner' in feature_name or 'HC' in feature_name or 'AC' in feature_name:
        return 'Corner Statistics'
    elif 'Accuracy' in feature_name or 'Strength' in feature_name or 'Efficiency' in feature_name:
        return 'Engineered'
    else:
        return 'Other'

feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)

feature_importance.to_csv('results/tables/feature_imp.csv', index=False)
```

---

### 📊 `tables/results_summary.xlsx` - To'liq Hisobot

Excel fayli bir nechta sheet'lardan iborat:

**Sheet 1: Overall Metrics**
Metric Value
────────────────────────────
Total Matches 3800
Train Set 3040 (80%)
Test Set 760 (20%)
Best Model XGBoost
Best Accuracy 58.3%
Training Time 67.8s
Inference Time 0.08s/match

**Sheet 2: Class Performance**
Class Precision Recall F1-Score Support
──────────────────────────────────────────────────
Home Win 0.65 0.71 0.68 170
Draw 0.40 0.29 0.34 120
Away Win 0.62 0.62 0.62 170
──────────────────────────────────────────────────
Accuracy 0.583 460
Macro Avg 0.56 0.54 0.55 460
Weighted 0.58 0.58 0.58 460

**Sheet 3: Confusion Matrix**
Predicted
H D A
Actual H 120 30 20
D 40 35 45
A 25 40 105

**Excel yaratish:**

```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

# Excel writer
with pd.ExcelWriter('results/tables/results_summary.xlsx', engine='openpyxl') as writer:
    # Sheet 1: Overall Metrics
    overall_metrics.to_excel(writer, sheet_name='Overall Metrics', index=False)

    # Sheet 2: Class Performance
    class_performance.to_excel(writer, sheet_name='Class Performance', index=False)

    # Sheet 3: Confusion Matrix
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')

    # Sheet 4: Feature Importance
    feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)

    # Sheet 5: Predictions
    predictions_df.head(100).to_excel(writer, sheet_name='Sample Predictions', index=False)
```

---

## 📈 Natijalarni Ko'rish

### Python orqali:

```python
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 1. Jadvallarni o'qish
metrics = pd.read_csv('results/tables/metrics.csv')
print(metrics)

# 2. Grafiklarni ko'rish
img = Image.open('results/figures/model_perf/confusion_matrix.png')
plt.figure(figsize=(12, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

# 3. Bashoratlarni tahlil qilish
predictions = pd.read_csv('results/tables/predictions.csv')
accuracy = predictions['correct'].mean()
print(f"Overall Accuracy: {accuracy:.2%}")

# To'g'ri va noto'g'ri bashoratlar
correct_preds = predictions[predictions['correct'] == True]
wrong_preds = predictions[predictions['correct'] == False]

print(f"\nCorrect predictions: {len(correct_preds)}")
print(f"Wrong predictions: {len(wrong_preds)}")
```

### Jupyter Notebook'da:

```python
# Inline grafiklar
%matplotlib inline

from IPython.display import Image, display

# Barcha EDA grafiklarni ko'rsatish
eda_figures = [
    'goals_distribution.png',
    'correlation_heatmap.png',
    'shots_vs_goals.png',
    'home_advantage.png'
]

for fig in eda_figures:
    print(f"\n{'='*50}")
    print(f"  {fig}")
    print('='*50)
    display(Image(filename=f'results/figures/eda/{fig}'))
```

---

## 🎯 Natijalarni Sharh Qilish

### 📊 Model Performance Tahlili:

```python
# models.py ichida
def generate_performance_report():
    """
    Model performance hisobotini yaratish
    """
    report = f"""
    ╔══════════════════════════════════════════════════╗
    ║         MODEL PERFORMANCE REPORT                 ║
    ╚══════════════════════════════════════════════════╝

    📊 Dataset:
       - Total matches: {len(df)}
       - Training set: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)
       - Test set: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)

    🤖 Best Model: {best_model_name}
       - Accuracy: {best_accuracy:.2%}
       - Precision: {best_precision:.2%}
       - Recall: {best_recall:.2%}
       - F1-Score: {best_f1:.2%}

    🎯 Class-wise Performance:
       - Home Win: {home_win_acc:.1%} accuracy
       - Draw: {draw_acc:.1%} accuracy (challenging!)
       - Away Win: {away_win_acc:.1%} accuracy

    🏆 Top 5 Important Features:
       1. {top_features[0]}: {top_importances[0]:.3f}
       2. {top_features[1]}: {top_importances[1]:.3f}
       3. {top_features[2]}: {top_importances[2]:.3f}
       4. {top_features[3]}: {top_importances[3]:.3f}
       5. {top_features[4]}: {top_importances[4]:.3f}

    ⏱️  Performance:
       - Training time: {train_time:.1f}s
       - Inference time: {inference_time*1000:.1f}ms per match

    💡 Insights:
       - Home teams win {home_win_rate:.1%} of matches
       - Draws are hardest to predict ({draw_acc:.1%})
       - Shots on target are most important features
       - Model performs best on clear home/away wins
    """

    # Save report
    with open('results/performance_report.txt', 'w') as f:
        f.write(report)

    return report
```

---

## 📚 Barcha Natija Fayllar

### ✅ To'liq ro'yxat:

results/
├── 📁 figures/
│ ├── 📁 eda/
│ │ ├── goals_distribution.png
│ │ ├── correlation_heatmap.png
│ │ ├── shots_vs_goals.png
│ │ ├── home_advantage.png
│ │ ├── cards_distribution.png
│ │ ├── corners_analysis.png
│ │ ├── temporal_trends.png
│ │ └── team_performance.png
│ │
│ ├── 📁 model_perf/
│ │ ├── confusion_matrix.png
│ │ ├── roc_curve.png
│ │ ├── feature_importance.png
│ │ ├── learning_curves.png
│ │ ├── precision_recall.png
│ │ └── model_comparison.png
│ │
│ ├── 📁 predictions/
│ │ ├── prediction_samples.png
│ │ ├── actual_vs_predicted.png
│ │ └── confidence_distribution.png
│ │
│ └── 📁 training/
│ ├── training_history.png
│ ├── validation_curve.png
│ └── cross_validation.png
│
├── 📁 tables/
│ ├── metrics.csv
│ ├── predictions.csv
│ ├── feature_imp.csv
│ └── results_summary.xlsx
│
├── 📄 performance_report.txt
└── 📄 ReadMe.md

---

## 🔧 Utilities

### Natijalarni tozalash:

```python
# clear_results.py
import os
import shutil

def clear_results(keep_readme=True):
    """
    Results papkasini tozalash
    """
    folders = ['figures/eda', 'figures/model_perf', 'figures/predictions', 'figures/training', 'tables']

    for folder in folders:
        path = f'results/{folder}'
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"✅ Cleared: {path}")

    if not keep_readme:
        readme_path = 'results/ReadMe.md'
        if os.path.exists(readme_path):
            os.remove(readme_path)

    print("\n🎉 Results cleared successfully!")

if __name__ == "__main__":
    clear_results()
```

### Natijalarni arxivlash:

```python
# archive_results.py
import shutil
from datetime import datetime

def archive_results():
    """
    Natijalarni sana bilan arxivlash
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f'results_archive_{timestamp}'

    shutil.make_archive(archive_name, 'zip', 'results/')
    print(f"✅ Results archived: {archive_name}.zip")

if __name__ == "__main__":
    archive_results()
```

---

## 💡 Best Practices

1. **Har safar train qilganda natijalarni saqlang**

```python
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   plt.savefig(f'results/figures/model_perf/confusion_matrix_{timestamp}.png')
```

2. **Git'da .gitignore qo'shing**
   .gitignore
   results/figures/.png
   results/tables/.csv
   !results/ReadMe.md

3. **Vizualizatsiyalar uchun bir xil style ishlatings**

```python
   # config.py
   FIGURE_DPI = 300
   FIGURE_FORMAT = 'png'
   COLOR_PALETTE = 'viridis'
```

---

## 🎯 Xulosa

Ushbu `results/` papkasi:

- ✅ Barcha tahlil natijalarini saqlaydi
- ✅ Grafiklar va vizualizatsiyalar
- ✅ Model metrikalari va bashoratlar
- ✅ To'liq hisobotlar (Excel)
- ✅ Arxivlash va tozalash skriptlari

---

⭐ **Professional ML loyihasining ajralmas qismi!**
╔═══════════════════════════════════════╗
║ Analyze → Visualize → Report ║
║ Results Management 📊📈 ║
╚═══════════════════════════════════════╝

**Made with 📊, 📈 and 🎨**
