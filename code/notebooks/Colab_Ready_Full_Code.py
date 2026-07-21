"""
Cell2Cell Telecom Churn — EDA & Baseline Models
MSc Dissertation: Rajarajachozhan V K (03189049)
Supervisor: Ibukun Afolabi

HOW TO RUN IN GOOGLE COLAB:
1. Upload cell2cell-duke_univeristy.csv when prompted
2. This script runs all phases automatically
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# UPLOAD DATASET (Colab only)
# ============================================================
from google.colab import files
print("Please upload cell2cell-duke_univeristy.csv")
uploaded = files.upload()

# ============================================================
# PHASE 1: DATA LOADING & INITIAL EXPLORATION
# ============================================================
print("=" * 60)
print("PHASE 1: DATA LOADING & INITIAL EXPLORATION")
print("=" * 60)

df = pd.read_csv("cell2cell-duke_univeristy.csv")

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nTarget variable (churn):")
print(f"  Non-churners (0): {(df['churn']==0).sum():,} ({(df['churn']==0).mean()*100:.1f}%)")
print(f"  Churners (1):     {(df['churn']==1).sum():,} ({(df['churn']==1).mean()*100:.1f}%)")

# Drop unnecessary columns
drop_cols = ['Unnamed: 0', 'X', 'customer', 'traintest', 'churndep']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"\nAfter dropping ID/metadata columns: {df.shape[1]} features remaining")

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(f"\nMissing values summary:")
print(f"  Total columns with missing data: {(missing > 0).sum()}")
for col in missing[missing > 0].sort_values(ascending=False).index:
    print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")

# Data types
print(f"\nAll features are numeric: {df.select_dtypes(include=[np.number]).shape[1]} numeric columns")

# Basic statistics
print(f"\nKey feature statistics:")
key_features = ['revenue', 'mou', 'months', 'custcare', 'retcalls', 'overage', 'recchrge']
for feat in key_features:
    if feat in df.columns:
        print(f"  {feat}: mean={df[feat].mean():.1f}, median={df[feat].median():.1f}, std={df[feat].std():.1f}")

# ============================================================
# PHASE 2: EDA VISUALISATIONS
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Cell2Cell Telecom Churn — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Churn distribution
churn_counts = df['churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0,0].bar(['Non-Churn (0)', 'Churn (1)'], churn_counts.values, color=colors, edgecolor='black')
axes[0,0].set_title('Target Variable Distribution', fontweight='bold')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(churn_counts.values):
    axes[0,0].text(i, v + 500, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')

# 2. Monthly revenue by churn
df.groupby('churn')['revenue'].plot(kind='kde', ax=axes[0,1], legend=True)
axes[0,1].set_title('Monthly Revenue Distribution by Churn', fontweight='bold')
axes[0,1].set_xlabel('Monthly Revenue ($)')
axes[0,1].legend(['Non-Churn', 'Churn'])
axes[0,1].set_xlim(-50, 200)

# 3. Months in service by churn
df.groupby('churn')['months'].plot(kind='kde', ax=axes[0,2], legend=True)
axes[0,2].set_title('Months in Service by Churn', fontweight='bold')
axes[0,2].set_xlabel('Months')
axes[0,2].legend(['Non-Churn', 'Churn'])

# 4. Customer care calls by churn
churn_groups = df.groupby('churn')['custcare'].mean()
axes[1,0].bar(['Non-Churn', 'Churn'], churn_groups.values, color=colors, edgecolor='black')
axes[1,0].set_title('Avg Customer Care Calls by Churn', fontweight='bold')
axes[1,0].set_ylabel('Mean Calls')

# 5. Overage minutes by churn
churn_overage = df.groupby('churn')['overage'].mean()
axes[1,1].bar(['Non-Churn', 'Churn'], churn_overage.values, color=colors, edgecolor='black')
axes[1,1].set_title('Avg Overage Minutes by Churn', fontweight='bold')
axes[1,1].set_ylabel('Mean Minutes')

# 6. Retention calls by churn
churn_ret = df.groupby('churn')['retcalls'].mean()
axes[1,2].bar(['Non-Churn', 'Churn'], churn_ret.values, color=colors, edgecolor='black')
axes[1,2].set_title('Avg Retention Calls by Churn', fontweight='bold')
axes[1,2].set_ylabel('Mean Calls')

plt.tight_layout()
plt.show()

# Correlation heatmap for top features
fig2, ax2 = plt.subplots(figsize=(14, 10))
top_features = ['churn', 'revenue', 'mou', 'recchrge', 'overage', 'months',
                'custcare', 'retcalls', 'dropvce', 'blckvce', 'unansvce',
                'outcalls', 'incalls', 'peakvce', 'opeakvce', 'eqpdays']
corr = df[top_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax2)
ax2.set_title('Correlation Heatmap — Key Features vs Churn', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Top correlations with churn
churn_corr = df.corr()['churn'].drop('churn').abs().sort_values(ascending=False)
print(f"\nTop 10 features correlated with churn:")
for feat, corr_val in churn_corr.head(10).items():
    direction = "+" if df.corr()['churn'][feat] > 0 else "-"
    print(f"  {feat}: {direction}{corr_val:.3f}")

# ============================================================
# PHASE 3: PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: PREPROCESSING")
print("=" * 60)

# Separate target
X = df.drop('churn', axis=1)
y = df['churn']

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]:,}")

# Impute missing values — median for all (all numeric)
missing_before = X.isnull().sum().sum()
X = X.fillna(X.median())
print(f"Missing values imputed: {missing_before:,} → {X.isnull().sum().sum()}")

# 70/30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]:,} samples")
print(f"Test set:  {X_test.shape[0]:,} samples")
print(f"Train churn rate: {y_train.mean()*100:.1f}%")
print(f"Test churn rate:  {y_test.mean()*100:.1f}%")

# Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
print(f"\nMin-Max scaling applied. Feature range: [0, 1]")

# SMOTE — apply on training data only
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
print(f"\nBefore SMOTE — Class 0: {(y_train==0).sum():,}, Class 1: {(y_train==1).sum():,}")
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE  — Class 0: {(y_train_resampled==0).sum():,}, Class 1: {(y_train_resampled==1).sum():,}")

# ============================================================
# PHASE 4: BASELINE MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: MODEL TRAINING (with SMOTE)")
print("=" * 60)

results = {}

# --- Model 1: Logistic Regression ---
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1': f1_score(y_test, y_pred_lr),
    'AUC-ROC': roc_auc_score(y_test, y_prob_lr)
}

for metric, val in results['Logistic Regression'].items():
    print(f"  {metric}: {val:.4f}")

# --- Model 2: Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1': f1_score(y_test, y_pred_rf),
    'AUC-ROC': roc_auc_score(y_test, y_prob_rf)
}

for metric, val in results['Random Forest'].items():
    print(f"  {metric}: {val:.4f}")

# --- Model 3: XGBoost ---
print("\n--- XGBoost ---")
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                        random_state=42, subsample=0.8, use_label_encoder=False,
                        eval_metric='logloss')
    xgb.fit(X_train_resampled, y_train_resampled)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_name = 'XGBoost'
except ImportError:
    print("  XGBoost not available, using GradientBoosting as proxy")
    xgb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    random_state=42, subsample=0.8)
    xgb.fit(X_train_resampled, y_train_resampled)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_name = 'Gradient Boosting'

results[xgb_name] = {
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1': f1_score(y_test, y_pred_xgb),
    'AUC-ROC': roc_auc_score(y_test, y_prob_xgb)
}

for metric, val in results[xgb_name].items():
    print(f"  {metric}: {val:.4f}")

# ============================================================
# PHASE 5: RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T
print("\n" + results_df.round(4).to_string())

# Find best model
best_model = results_df['AUC-ROC'].idxmax()
print(f"\nBest model by AUC-ROC: {best_model} ({results_df.loc[best_model, 'AUC-ROC']:.4f})")

# ============================================================
# PHASE 6: VISUALISE RESULTS
# ============================================================

# ROC curves
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

model_colors = ['#3498db', '#2ecc71', '#e74c3c']
for (name, y_prob), color in zip([('Logistic Regression', y_prob_lr),
                                   ('Random Forest', y_prob_rf),
                                   (xgb_name, y_prob_xgb)], model_colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes3[0].plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

axes3[0].plot([0,1], [0,1], 'k--', alpha=0.3)
axes3[0].set_xlabel('False Positive Rate')
axes3[0].set_ylabel('True Positive Rate')
axes3[0].set_title('ROC Curves — Model Comparison', fontweight='bold')
axes3[0].legend()

# Model comparison bar chart
x_pos = np.arange(len(results_df.columns))
width = 0.25
for i, (model, color) in enumerate(zip(results_df.index, model_colors)):
    axes3[1].bar(x_pos + i*width, results_df.loc[model].values, width, label=model, color=color, edgecolor='black')

axes3[1].set_xticks(x_pos + width)
axes3[1].set_xticklabels(results_df.columns, rotation=15)
axes3[1].set_title('Model Performance Comparison', fontweight='bold')
axes3[1].set_ylim(0, 1)
axes3[1].legend()

plt.tight_layout()
plt.show()

# Confusion matrices
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, y_pred) in zip(axes4, [('Logistic Regression', y_pred_lr),
                                       ('Random Forest', y_pred_rf),
                                       (xgb_name, y_pred_xgb)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Churn', 'Churn'], yticklabels=['Non-Churn', 'Churn'])
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Feature importance from Random Forest
fig5, ax5 = plt.subplots(figsize=(12, 8))
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.tail(15).plot(kind='barh', ax=ax5, color='#3498db', edgecolor='black')
ax5.set_title('Top 15 Features — Random Forest Importance', fontsize=14, fontweight='bold')
ax5.set_xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Dataset: Cell2Cell (Duke University), 71,047 records, 67 features after cleanup
Churn rate: 29.0% (class imbalance present — SMOTE applied)
Preprocessing: median imputation, min-max scaling, 70/30 stratified split

Model results (with SMOTE):
{results_df.round(4).to_string()}

Best model: {best_model} (AUC-ROC: {results_df.loc[best_model, 'AUC-ROC']:.4f})

Next steps:
  1. Grid search hyperparameter tuning with 5-fold CV
  2. Apply SHAP and LIME to best model
  3. Four-metric evaluation (fidelity, stability, sparsity, agreement)
  4. SHAP-vector k-means clustering for churn segmentation
  5. Build Streamlit dashboard
""")

print("DONE — All phases complete.")
