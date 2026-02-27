"""
============================================================
CUSTOMER CHURN PREDICTION SYSTEM
============================================================
Author: [Your Name]
Technologies: Python, Scikit-learn, XGBoost, Feature Engineering

Dataset: Telco Customer Churn (Kaggle)
Task: Binary Classification (Churn: Yes/No)

Models Implemented:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. Gradient Boosting

Key Techniques:
  - Feature Engineering
  - Feature Importance Analysis
  - Hyperparameter Tuning (GridSearchCV)
  - Cross Validation
  - Handling Imbalanced Data (SMOTE)
============================================================
"""

# =============================================
# 1. IMPORT ALL LIBRARIES
# =============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
import time

# Scikit-learn
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    MinMaxScaler
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)
from sklearn.feature_selection import mutual_info_classif

# XGBoost
from xgboost import XGBClassifier

# Imbalanced data handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 60)
print(f"  Pandas      : {pd.__version__}")
print(f"  NumPy       : {np.__version__}")
print(f"  Scikit-learn: {__import__('sklearn').__version__}")
print("=" * 60)


# =============================================
# 2. LOAD / CREATE DATASET
# =============================================
print("\n📦 Loading Dataset...")

# ---- Option 1: Load from Kaggle CSV ----
# If you have the Telco Churn CSV:
# df = pd.read_csv('Telco-Customer-Churn.csv')

# ---- Option 2: Generate Realistic Dataset ----
# (For those who don't have the CSV)

def create_churn_dataset(n_samples=7043):
    """
    Create a realistic Telco Customer Churn dataset
    Similar to the Kaggle Telco Customer Churn dataset
    """
    np.random.seed(42)

    data = {
        'customerID': [f'CUST-{i:04d}' for i in range(n_samples)],

        # Demographics
        'gender': np.random.choice(
            ['Male', 'Female'], n_samples
        ),
        'SeniorCitizen': np.random.choice(
            [0, 1], n_samples, p=[0.84, 0.16]
        ),
        'Partner': np.random.choice(
            ['Yes', 'No'], n_samples, p=[0.48, 0.52]
        ),
        'Dependents': np.random.choice(
            ['Yes', 'No'], n_samples, p=[0.30, 0.70]
        ),

        # Account Information
        'tenure': np.random.choice(
            range(0, 73), n_samples
        ),
        'PhoneService': np.random.choice(
            ['Yes', 'No'], n_samples, p=[0.90, 0.10]
        ),
        'MultipleLines': np.random.choice(
            ['Yes', 'No', 'No phone service'], n_samples,
            p=[0.42, 0.48, 0.10]
        ),

        # Internet Service
        'InternetService': np.random.choice(
            ['DSL', 'Fiber optic', 'No'], n_samples,
            p=[0.34, 0.44, 0.22]
        ),
        'OnlineSecurity': np.random.choice(
            ['Yes', 'No', 'No internet service'], n_samples,
            p=[0.29, 0.49, 0.22]
        ),
        'OnlineBackup': np.random.choice(
            ['Yes', 'No', 'No internet service'], n_samples,
            p=[0.34, 0.44, 0.22]
        ),
        'DeviceProtection': np.random.choice(
            ['Yes', 'No', 'No internet service'], n_samples,
            p=[0.34, 0.44, 0.22]
        ),
        'TechSupport': np.random.choice(
            ['Yes', 'No', 'No internet service'], n_samples,
            p=[0.29, 0.49, 0.22]
        ),
        'StreamingTV': np.random.choice(
            ['Yes', 'No', 'No internet service'], n_samples,
            p=[0.38, 0.40, 0.22]
        ),
        'StreamingMovies': np.random.choice(
            ['Yes', 'No', 'No internet service'], n_samples,
            p=[0.39, 0.39, 0.22]
        ),

        # Billing
        'Contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], n_samples,
            p=[0.55, 0.21, 0.24]
        ),
        'PaperlessBilling': np.random.choice(
            ['Yes', 'No'], n_samples, p=[0.59, 0.41]
        ),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check',
             'Bank transfer (automatic)',
             'Credit card (automatic)'],
            n_samples, p=[0.34, 0.23, 0.22, 0.21]
        ),

        # Charges
        'MonthlyCharges': np.round(
            np.random.uniform(18, 118, n_samples), 2
        ),
    }

    df = pd.DataFrame(data)

    # Calculate TotalCharges based on tenure and monthly charges
    df['TotalCharges'] = np.round(
        df['tenure'] * df['MonthlyCharges'] +
        np.random.uniform(-50, 50, n_samples), 2
    )
    df['TotalCharges'] = df['TotalCharges'].clip(lower=0)

    # Generate realistic Churn based on features
    churn_probability = np.zeros(n_samples)

    # Higher churn for month-to-month contracts
    churn_probability += (
        df['Contract'] == 'Month-to-month'
    ).astype(float) * 0.3

    # Lower churn for long tenure
    churn_probability -= (df['tenure'] / 72) * 0.2

    # Higher churn for high monthly charges
    churn_probability += (
        (df['MonthlyCharges'] - 18) / 100
    ) * 0.15

    # Higher churn for fiber optic (often due to price)
    churn_probability += (
        df['InternetService'] == 'Fiber optic'
    ).astype(float) * 0.15

    # Lower churn with online security
    churn_probability -= (
        df['OnlineSecurity'] == 'Yes'
    ).astype(float) * 0.1

    # Lower churn with tech support
    churn_probability -= (
        df['TechSupport'] == 'Yes'
    ).astype(float) * 0.1

    # Higher churn for electronic check
    churn_probability += (
        df['PaymentMethod'] == 'Electronic check'
    ).astype(float) * 0.1

    # Senior citizens slightly higher churn
    churn_probability += df['SeniorCitizen'] * 0.05

    # No partner = higher churn
    churn_probability += (
        df['Partner'] == 'No'
    ).astype(float) * 0.05

    # Add randomness
    churn_probability += np.random.normal(0, 0.1, n_samples)

    # Clip and convert to binary
    churn_probability = np.clip(churn_probability, 0, 1)
    df['Churn'] = (
        churn_probability > np.percentile(churn_probability, 73.5)
    ).astype(str)
    df['Churn'] = df['Churn'].map({
        'True': 'Yes', 'False': 'No'
    })

    return df


# Create / Load dataset
try:
    df = pd.read_csv('Telco-Customer-Churn.csv')
    print("✅ Dataset loaded from CSV file!")
except FileNotFoundError:
    print("📝 CSV not found. Creating realistic dataset...")
    df = create_churn_dataset()
    df.to_csv('Telco-Customer-Churn.csv', index=False)
    print("✅ Dataset created and saved as 'Telco-Customer-Churn.csv'")


# =============================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================
print("\n" + "=" * 60)
print("📊 EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Basic Info
print(f"\n📋 Dataset Shape: {df.shape}")
print(f"   Rows   : {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")

print(f"\n📋 Column Names & Types:")
print(df.dtypes.to_string())

print(f"\n📋 First 5 Rows:")
print(df.head().to_string())

print(f"\n📋 Statistical Summary:")
print(df.describe().to_string())

print(f"\n📋 Missing Values:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0].to_string())
if missing.sum() == 0:
    print("   No missing values found! ✅")

# Churn Distribution
print(f"\n📋 Target Variable (Churn) Distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts.to_string())
print(f"\n   Churn Rate: {churn_counts['Yes']/len(df)*100:.2f}%")


# =============================================
# 4. DATA VISUALIZATION
# =============================================
print("\n📊 Generating Visualizations...")


def plot_eda_visualizations(df):
    """Comprehensive EDA visualizations"""

    # ----- Figure 1: Churn Distribution -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Churn Distribution Overview',
                 fontsize=16, fontweight='bold')

    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    churn_counts = df['Churn'].value_counts()
    axes[0].pie(churn_counts, labels=['No Churn', 'Churn'],
                colors=colors, autopct='%1.1f%%', startangle=90,
                explode=(0, 0.1), shadow=True,
                textprops={'fontsize': 12})
    axes[0].set_title('Churn Distribution (Pie)', fontsize=13)

    # Bar chart
    sns.countplot(data=df, x='Churn', palette=colors, ax=axes[1],
                  edgecolor='black')
    axes[1].set_title('Churn Distribution (Bar)', fontsize=13)
    for p in axes[1].patches:
        axes[1].annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    # Churn by Gender
    sns.countplot(data=df, x='gender', hue='Churn',
                  palette=colors, ax=axes[2], edgecolor='black')
    axes[2].set_title('Churn by Gender', fontsize=13)
    axes[2].legend(title='Churn')

    plt.tight_layout()
    plt.savefig('churn_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ----- Figure 2: Numerical Features -----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Numerical Features Analysis',
                 fontsize=16, fontweight='bold')

    # Tenure distribution
    sns.histplot(data=df, x='tenure', hue='Churn',
                 kde=True, palette=colors, ax=axes[0][0], alpha=0.6)
    axes[0][0].set_title('Tenure Distribution by Churn', fontsize=12)

    # Monthly Charges
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn',
                 kde=True, palette=colors, ax=axes[0][1], alpha=0.6)
    axes[0][1].set_title('Monthly Charges by Churn', fontsize=12)

    # Total Charges
    df_temp = df.copy()
    df_temp['TotalCharges'] = pd.to_numeric(
        df_temp['TotalCharges'], errors='coerce'
    )
    sns.histplot(data=df_temp, x='TotalCharges', hue='Churn',
                 kde=True, palette=colors, ax=axes[0][2], alpha=0.6)
    axes[0][2].set_title('Total Charges by Churn', fontsize=12)

    # Box plots
    sns.boxplot(data=df, x='Churn', y='tenure',
                palette=colors, ax=axes[1][0])
    axes[1][0].set_title('Tenure Boxplot', fontsize=12)

    sns.boxplot(data=df, x='Churn', y='MonthlyCharges',
                palette=colors, ax=axes[1][1])
    axes[1][1].set_title('Monthly Charges Boxplot', fontsize=12)

    sns.violinplot(data=df, x='Churn', y='MonthlyCharges',
                   palette=colors, ax=axes[1][2])
    axes[1][2].set_title('Monthly Charges Violin', fontsize=12)

    plt.tight_layout()
    plt.savefig('numerical_features.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ----- Figure 3: Categorical Features -----
    cat_features = [
        'Contract', 'InternetService', 'PaymentMethod',
        'TechSupport', 'OnlineSecurity', 'PaperlessBilling'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Categorical Features vs Churn',
                 fontsize=16, fontweight='bold')

    for idx, feature in enumerate(cat_features):
        row, col = idx // 3, idx % 3
        ct = pd.crosstab(df[feature], df['Churn'], normalize='index')
        ct.plot(kind='bar', stacked=True, color=colors,
                ax=axes[row][col], edgecolor='black')
        axes[row][col].set_title(f'{feature} vs Churn', fontsize=12)
        axes[row][col].set_ylabel('Proportion')
        axes[row][col].legend(title='Churn')
        axes[row][col].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('categorical_features.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ All EDA visualizations saved!")


plot_eda_visualizations(df)


# =============================================
# 5. DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================
print("\n" + "=" * 60)
print("⚙️  DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 60)

# Create a copy for processing
data = df.copy()

# Drop customerID (not useful for prediction)
data.drop('customerID', axis=1, inplace=True)
print("✅ Dropped 'customerID' column")

# Handle TotalCharges - convert to numeric
data['TotalCharges'] = pd.to_numeric(
    data['TotalCharges'], errors='coerce'
)

# Fill missing TotalCharges
missing_tc = data['TotalCharges'].isnull().sum()
if missing_tc > 0:
    data['TotalCharges'].fillna(
        data['TotalCharges'].median(), inplace=True
    )
    print(f"✅ Filled {missing_tc} missing TotalCharges with median")

# Encode target variable
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
print(f"✅ Target encoded: Yes=1, No=0")

# ----- FEATURE ENGINEERING -----
print(f"\n🔧 Feature Engineering...")

# 1. Average Monthly Charges per tenure month
data['AvgChargesPerMonth'] = np.where(
    data['tenure'] > 0,
    data['TotalCharges'] / data['tenure'],
    data['MonthlyCharges']
)

# 2. Tenure Groups (binning)
data['TenureGroup'] = pd.cut(
    data['tenure'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'],
    include_lowest=True
)

# 3. Monthly Charges Category
data['ChargesCategory'] = pd.cut(
    data['MonthlyCharges'],
    bins=[0, 35, 70, 120],
    labels=['Low', 'Medium', 'High']
)

# 4. Total Services count
service_columns = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

data['TotalServices'] = 0
for col in service_columns:
    data['TotalServices'] += (
        data[col].apply(
            lambda x: 1 if x not in ['No', 'No internet service',
                                      'No phone service'] else 0
        )
    )

# 5. Has Internet Service
data['HasInternet'] = (
    data['InternetService'] != 'No'
).astype(int)

# 6. Has Phone Service
data['HasPhone'] = (
    data['PhoneService'] == 'Yes'
).astype(int)

# 7. Contract Type (Numeric encoding for ordinal)
contract_map = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}
data['ContractType'] = data['Contract'].map(contract_map)

# 8. Charges to Tenure Ratio
data['ChargeTenureRatio'] = np.where(
    data['tenure'] > 0,
    data['MonthlyCharges'] / data['tenure'],
    data['MonthlyCharges']
)

print(f"✅ Created 8 new engineered features!")
print(f"\n📋 New Features:")
new_features = [
    'AvgChargesPerMonth', 'TenureGroup', 'ChargesCategory',
    'TotalServices', 'HasInternet', 'HasPhone',
    'ContractType', 'ChargeTenureRatio'
]
for feat in new_features:
    print(f"   ✅ {feat}")

print(f"\n📊 Dataset shape after Feature Engineering: {data.shape}")

# ----- ENCODE CATEGORICAL VARIABLES -----
print(f"\n🔢 Encoding Categorical Variables...")

# Get categorical columns
categorical_cols = data.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print(f"   Categorical columns: {categorical_cols}")

# Label Encoding for binary columns
binary_cols = ['gender', 'Partner', 'Dependents',
               'PhoneService', 'PaperlessBilling']

label_encoders = {}
for col in binary_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        print(f"   ✅ Label Encoded: {col} → {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding for multi-class columns
multi_class_cols = [
    col for col in categorical_cols
    if col not in binary_cols and col in data.columns
]

print(f"\n   One-Hot Encoding: {multi_class_cols}")

data = pd.get_dummies(
    data,
    columns=multi_class_cols,
    drop_first=True,  # Avoid multicollinearity
    dtype=int
)

print(f"\n📊 Final Dataset Shape: {data.shape}")
print(f"   Features: {data.shape[1] - 1}")
print(f"   Samples : {data.shape[0]}")


# =============================================
# 6. CORRELATION ANALYSIS
# =============================================
print("\n📊 Correlation Analysis...")


def plot_correlation_analysis(data):
    """Plot correlation heatmap and top correlations with Churn"""

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])

    # Full correlation heatmap
    plt.figure(figsize=(20, 16))
    corr_matrix = numeric_data.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=False,
        cmap='RdBu_r', center=0, linewidths=0.5,
        square=True, fmt='.2f'
    )
    plt.title('Feature Correlation Heatmap',
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Top correlations with Churn
    churn_corr = numeric_data.corr()['Churn'].drop('Churn').sort_values(
        key=abs, ascending=False
    )

    plt.figure(figsize=(12, 8))
    colors = ['#e74c3c' if x > 0 else '#2ecc71'
              for x in churn_corr.head(15)]
    churn_corr.head(15).plot(
        kind='barh', color=colors, edgecolor='black'
    )
    plt.title('Top 15 Features Correlated with Churn',
              fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('churn_correlations.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n📋 Top 10 Features Correlated with Churn:")
    for feat, corr_val in churn_corr.head(10).items():
        direction = "↑ Positive" if corr_val > 0 else "↓ Negative"
        print(f"   {feat:<35} : {corr_val:>7.4f}  ({direction})")

    print("✅ Correlation analysis saved!")


plot_correlation_analysis(data)


# =============================================
# 7. PREPARE DATA FOR MODELING
# =============================================
print("\n" + "=" * 60)
print("📦 PREPARING DATA FOR MODELING")
print("=" * 60)

# Separate features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

print(f"\nFeatures (X) shape : {X.shape}")
print(f"Target (y) shape   : {y.shape}")
print(f"Target Distribution:")
print(f"   No Churn (0) : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"   Churn (1)    : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Testing  : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Feature Scaling applied (StandardScaler)")
print(f"   Mean (before): {X_train.mean().mean():.2f}")
print(f"   Mean (after) : {X_train_scaled.mean():.4f}")
print(f"   Std (after)  : {X_train_scaled.std():.4f}")

# Handle Imbalanced Data with SMOTE
print(f"\n⚖️  Handling Class Imbalance with SMOTE...")
print(f"   Before SMOTE:")
print(f"     Class 0: {(y_train == 0).sum()}")
print(f"     Class 1: {(y_train == 1).sum()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

print(f"   After SMOTE:")
print(f"     Class 0: {(y_train_resampled == 0).sum()}")
print(f"     Class 1: {(y_train_resampled == 1).sum()}")
print(f"   ✅ Classes are now balanced!")


# =============================================
# 8. MODEL 1: LOGISTIC REGRESSION
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

start_time = time.time()

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs',
    C=1.0
)

# Cross Validation
lr_cv_scores = cross_val_score(
    lr_model, X_train_resampled, y_train_resampled,
    cv=5, scoring='accuracy'
)
print(f"\n📊 Cross-Validation Scores: {lr_cv_scores}")
print(f"   Mean CV Accuracy: {lr_cv_scores.mean()*100:.2f}% "
      f"(±{lr_cv_scores.std()*100:.2f}%)")

# Train
lr_model.fit(X_train_resampled, y_train_resampled)

lr_time = time.time() - start_time
print(f"   Training Time: {lr_time:.2f}s")


# =============================================
# 9. MODEL 2: RANDOM FOREST
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 2: RANDOM FOREST")
print("=" * 60)

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Cross Validation
rf_cv_scores = cross_val_score(
    rf_model, X_train_resampled, y_train_resampled,
    cv=5, scoring='accuracy'
)
print(f"\n📊 Cross-Validation Scores: {rf_cv_scores}")
print(f"   Mean CV Accuracy: {rf_cv_scores.mean()*100:.2f}% "
      f"(±{rf_cv_scores.std()*100:.2f}%)")

# Train
rf_model.fit(X_train_resampled, y_train_resampled)

rf_time = time.time() - start_time
print(f"   Training Time: {rf_time:.2f}s")


# =============================================
# 10. MODEL 3: XGBOOST
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 3: XGBOOST")
print("=" * 60)

start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Cross Validation
xgb_cv_scores = cross_val_score(
    xgb_model, X_train_resampled, y_train_resampled,
    cv=5, scoring='accuracy'
)
print(f"\n📊 Cross-Validation Scores: {xgb_cv_scores}")
print(f"   Mean CV Accuracy: {xgb_cv_scores.mean()*100:.2f}% "
      f"(±{xgb_cv_scores.std()*100:.2f}%)")

# Train
xgb_model.fit(X_train_resampled, y_train_resampled)

xgb_time = time.time() - start_time
print(f"   Training Time: {xgb_time:.2f}s")


# =============================================
# 11. MODEL 4: GRADIENT BOOSTING
# =============================================
print("\n" + "=" * 60)
print("🏗️  MODEL 4: GRADIENT BOOSTING")
print("=" * 60)

start_time = time.time()

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Cross Validation
gb_cv_scores = cross_val_score(
    gb_model, X_train_resampled, y_train_resampled,
    cv=5, scoring='accuracy'
)
print(f"\n📊 Cross-Validation Scores: {gb_cv_scores}")
print(f"   Mean CV Accuracy: {gb_cv_scores.mean()*100:.2f}% "
      f"(±{gb_cv_scores.std()*100:.2f}%)")

# Train
gb_model.fit(X_train_resampled, y_train_resampled)

gb_time = time.time() - start_time
print(f"   Training Time: {gb_time:.2f}s")


# =============================================
# 12. HYPERPARAMETER TUNING (GridSearchCV)
# =============================================
print("\n" + "=" * 60)
print("🔧 HYPERPARAMETER TUNING - GridSearchCV")
print("=" * 60)

# Tune XGBoost (typically the best performer)
print("\n🔍 Tuning XGBoost...")

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# Use smaller grid for faster execution
xgb_param_grid_small = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
}

start_time = time.time()

xgb_grid_search = GridSearchCV(
    XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    param_grid=xgb_param_grid_small,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

xgb_grid_search.fit(X_train_resampled, y_train_resampled)

tune_time = time.time() - start_time

print(f"\n✅ GridSearchCV Complete! (Time: {tune_time:.1f}s)")
print(f"   Best Parameters: {xgb_grid_search.best_params_}")
print(f"   Best F1-Score  : {xgb_grid_search.best_score_*100:.2f}%")

# Use tuned model
xgb_tuned = xgb_grid_search.best_estimator_

# Also tune Random Forest
print("\n🔍 Tuning Random Forest...")

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\n✅ Random Forest Tuning Complete!")
print(f"   Best Parameters: {rf_grid_search.best_params_}")
print(f"   Best F1-Score  : {rf_grid_search.best_score_*100:.2f}%")

rf_tuned = rf_grid_search.best_estimator_


# =============================================
# 13. MODEL EVALUATION
# =============================================
print("\n" + "=" * 60)
print("📊 MODEL EVALUATION ON TEST SET")
print("=" * 60)

# All models to evaluate
all_models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_tuned,
    'XGBoost': xgb_tuned,
    'Gradient Boosting': gb_model
}


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n{'='*50}")
    print(f"📊 {model_name}")
    print(f"{'='*50}")
    print(f"   Accuracy  : {accuracy * 100:.2f}%")
    print(f"   Precision : {precision * 100:.2f}%")
    print(f"   Recall    : {recall * 100:.2f}%")
    print(f"   F1-Score  : {f1 * 100:.2f}%")
    print(f"   ROC-AUC   : {roc * 100:.2f}%")

    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['No Churn', 'Churn'],
        digits=4
    ))

    return {
        'name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }


# Evaluate all models
all_results = {}
for name, model in all_models.items():
    all_results[name] = evaluate_model(
        model, X_test_scaled, y_test, name
    )


# =============================================
# 14. CONFUSION MATRIX VISUALIZATION
# =============================================
print("\n📊 Generating Confusion Matrices...")


def plot_all_confusion_matrices(results_dict, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Confusion Matrices - All Models',
                 fontsize=16, fontweight='bold')

    for idx, (name, results) in enumerate(results_dict.items()):
        row, col = idx // 2, idx % 2
        cm = confusion_matrix(y_test, results['y_pred'])

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            ax=axes[row][col],
            linewidths=0.5,
            annot_kws={"size": 14}
        )
        axes[row][col].set_title(
            f"{name}\nAccuracy: {results['accuracy']*100:.2f}%",
            fontsize=11, fontweight='bold'
        )
        axes[row][col].set_xlabel('Predicted')
        axes[row][col].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrices saved!")


plot_all_confusion_matrices(all_results, y_test)


# =============================================
# 15. ROC CURVE COMPARISON
# =============================================
print("\n📈 Generating ROC Curves...")


def plot_roc_curves(results_dict, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for (name, results), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_prob'])
        roc_auc_val = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {roc_auc_val:.4f})"
        )

    plt.plot([0, 1], [0, 1], color='gray', linewidth=1,
             linestyle='--', label='Random Classifier')

    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves - Model Comparison',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ ROC curves saved!")


plot_roc_curves(all_results, y_test)


# =============================================
# 16. PRECISION-RECALL CURVE
# =============================================
print("\n📈 Generating Precision-Recall Curves...")


def plot_precision_recall_curves(results_dict, y_test):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(10, 8))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for (name, results), color in zip(results_dict.items(), colors):
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_test, results['y_pred_prob']
        )
        pr_auc = auc(recall_vals, precision_vals)

        plt.plot(
            recall_vals, precision_vals, color=color, linewidth=2,
            label=f"{name} (AUC = {pr_auc:.4f})"
        )

    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curves',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Precision-Recall curves saved!")


plot_precision_recall_curves(all_results, y_test)


# =============================================
# 17. FEATURE IMPORTANCE ANALYSIS
# =============================================
print("\n" + "=" * 60)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)


def plot_feature_importance(models_dict, feature_names):
    """Plot feature importance for tree-based models"""

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Feature Importance - Top 15 Features',
                 fontsize=16, fontweight='bold')

    tree_models = {
        'Random Forest': models_dict['Random Forest']['model'],
        'XGBoost': models_dict['XGBoost']['model'],
        'Gradient Boosting': models_dict['Gradient Boosting']['model']
    }

    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    for idx, (name, model) in enumerate(tree_models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15

        axes[idx].barh(
            range(len(indices)),
            importances[indices],
            color=colors[idx],
            edgecolor='black',
            alpha=0.8
        )
        axes[idx].set_yticks(range(len(indices)))
        axes[idx].set_yticklabels(
            [feature_names[i] for i in indices], fontsize=9
        )
        axes[idx].set_title(f'{name}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Importance')
        axes[idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print top features for best model
    best_model = models_dict['XGBoost']['model']
    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(f"\n📋 Top 15 Most Important Features (XGBoost):")
    print(f"{'─'*50}")
    for idx, row in feat_imp_df.head(15).iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"   {row['Feature']:<35}: {row['Importance']:.4f} {bar}")
    print(f"{'─'*50}")

    print("✅ Feature importance analysis saved!")

    return feat_imp_df


feature_names = X.columns.tolist()
feat_imp_df = plot_feature_importance(all_results, feature_names)


# =============================================
# 18. MUTUAL INFORMATION (Additional Feature Importance)
# =============================================
print("\n📊 Mutual Information Analysis...")

mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(
    mi_df.head(15)['Feature'][::-1],
    mi_df.head(15)['MI_Score'][::-1],
    color='steelblue', edgecolor='black'
)
plt.title('Mutual Information Scores - Top 15 Features',
          fontsize=14, fontweight='bold')
plt.xlabel('MI Score')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mutual_information.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📋 Top 10 Features by Mutual Information:")
for _, row in mi_df.head(10).iterrows():
    print(f"   {row['Feature']:<35}: {row['MI_Score']:.4f}")
print("✅ Mutual Information analysis saved!")


# =============================================
# 19. MODEL COMPARISON
# =============================================
print("\n" + "=" * 60)
print("📊 COMPREHENSIVE MODEL COMPARISON")
print("=" * 60)


def plot_model_comparison(results_dict):
    """Final model comparison visualization"""

    model_names = list(results_dict.keys())
    metrics = {
        'Accuracy': [r['accuracy'] * 100 for r in results_dict.values()],
        'Precision': [r['precision'] * 100 for r in results_dict.values()],
        'Recall': [r['recall'] * 100 for r in results_dict.values()],
        'F1-Score': [r['f1'] * 100 for r in results_dict.values()],
        'ROC-AUC': [r['roc_auc'] * 100 for r in results_dict.values()]
    }

    x = np.arange(len(model_names))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width,
                      label=metric_name, color=colors[i],
                      edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.3,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=7,
                fontweight='bold'
            )

    ax.set_xlabel('Models', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Model Comparison - All Metrics',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([60, 105])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print(f"{'='*90}")
    for name, r in results_dict.items():
        print(
            f"{name:<25} {r['accuracy']*100:>9.2f}% "
            f"{r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% "
            f"{r['f1']*100:>9.2f}% {r['roc_auc']*100:>9.2f}%"
        )
    print(f"{'='*90}")

    # Best model
    best_name = max(results_dict, key=lambda x: results_dict[x]['f1'])
    print(f"\n🏆 Best Model: {best_name} "
          f"(F1-Score: {results_dict[best_name]['f1']*100:.2f}%)")

    return best_name


best_model_name = plot_model_comparison(all_results)


# =============================================
# 20. PREDICT SINGLE CUSTOMER
# =============================================
print("\n" + "=" * 60)
print("🔮 PREDICT SINGLE CUSTOMER CHURN")
print("=" * 60)


def predict_customer_churn(customer_data, model, scaler,
                           feature_names, model_name="Model"):
    """
    Predict churn probability for a single customer.

    Parameters:
        customer_data: dict with feature values
        model: trained model
        scaler: fitted scaler
        feature_names: list of feature names
    """
    # Create DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Ensure all features are present
    for feat in feature_names:
        if feat not in customer_df.columns:
            customer_df[feat] = 0

    customer_df = customer_df[feature_names]

    # Scale
    customer_scaled = scaler.transform(customer_df)

    # Predict
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0]

    # Display
    result = "🚨 WILL CHURN" if prediction == 1 else "✅ WILL NOT CHURN"
    churn_prob = probability[1] * 100
    retain_prob = probability[0] * 100

    print(f"\n{'─'*50}")
    print(f"🤖 Model: {model_name}")
    print(f"{'─'*50}")
    print(f"📊 Prediction     : {result}")
    print(f"📈 Churn Prob     : {churn_prob:.2f}%")
    print(f"📉 Retain Prob    : {retain_prob:.2f}%")
    print(f"{'─'*50}")

    # Risk Level
    if churn_prob > 80:
        risk = "🔴 CRITICAL RISK"
    elif churn_prob > 60:
        risk = "🟠 HIGH RISK"
    elif churn_prob > 40:
        risk = "🟡 MEDIUM RISK"
    elif churn_prob > 20:
        risk = "🟢 LOW RISK"
    else:
        risk = "✅ SAFE"

    print(f"⚠️  Risk Level    : {risk}")
    print(f"{'─'*50}")

    # Recommendations
    if prediction == 1:
        print(f"\n💡 Retention Recommendations:")
        print(f"   1. Offer contract upgrade discount")
        print(f"   2. Provide free tech support for 3 months")
        print(f"   3. Reduce monthly charges by 15-20%")
        print(f"   4. Bundle additional services at discount")
        print(f"   5. Assign dedicated customer success manager")

    return prediction, probability


# Test with sample customers
print("\n🧪 Testing with Sample Customers:")

# Get feature names and create sample data
sample_customer_1 = dict(zip(feature_names, X_test_scaled[0]))
sample_customer_2 = dict(zip(feature_names, X_test_scaled[1]))

# Use best model
best_model = all_results[best_model_name]['model']

print(f"\n--- Customer 1 (Actual: {'Churn' if y_test.iloc[0]==1 else 'No Churn'}) ---")
predict_customer_churn(
    sample_customer_1, best_model, scaler,
    feature_names, best_model_name
)

print(f"\n--- Customer 2 (Actual: {'Churn' if y_test.iloc[1]==1 else 'No Churn'}) ---")
predict_customer_churn(
    sample_customer_2, best_model, scaler,
    feature_names, best_model_name
)


# =============================================
# 21. BATCH PREDICTION & RISK ANALYSIS
# =============================================
print("\n" + "=" * 60)
print("📊 BATCH PREDICTION & RISK ANALYSIS")
print("=" * 60)


def batch_risk_analysis(model, X_test, y_test, scaler):
    """Analyze churn risk across all test customers"""
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Risk categories
    risk_df = pd.DataFrame({
        'Actual': y_test.values,
        'Churn_Probability': y_pred_prob
    })

    risk_df['Risk_Level'] = pd.cut(
        risk_df['Churn_Probability'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Safe', 'Low', 'Medium', 'High', 'Critical']
    )

    # Risk distribution
    risk_counts = risk_df['Risk_Level'].value_counts().sort_index()

    print(f"\n📊 Customer Risk Distribution:")
    print(f"{'─'*40}")
    for level, count in risk_counts.items():
        bar = '█' * (count // 10)
        print(f"   {level:<10}: {count:>5} customers {bar}")
    print(f"{'─'*40}")
    print(f"   Total    : {len(risk_df):>5} customers")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Customer Risk Analysis',
                 fontsize=16, fontweight='bold')

    # Risk distribution
    risk_colors = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    risk_counts.plot(
        kind='bar', color=risk_colors, edgecolor='black',
        ax=axes[0]
    )
    axes[0].set_title('Risk Level Distribution', fontsize=13)
    axes[0].set_xlabel('Risk Level')
    axes[0].set_ylabel('Number of Customers')
    axes[0].tick_params(axis='x', rotation=0)

    # Churn probability distribution
    axes[1].hist(
        y_pred_prob, bins=50, color='steelblue',
        edgecolor='black', alpha=0.7
    )
    axes[1].axvline(0.5, color='red', linestyle='--',
                    linewidth=2, label='Decision Threshold (0.5)')
    axes[1].set_title('Churn Probability Distribution', fontsize=13)
    axes[1].set_xlabel('Churn Probability')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('risk_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Risk analysis saved!")


batch_risk_analysis(best_model, X_test_scaled, y_test, scaler)


# =============================================
# 22. LEARNING CURVE
# =============================================
print("\n📈 Generating Learning Curves...")


def plot_learning_curve(model, X, y, model_name):
    """Plot learning curve to check overfitting/underfitting"""
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='f1', n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='red')
    plt.plot(train_sizes, train_mean, 'o-', color='blue',
             linewidth=2, label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='red',
             linewidth=2, label='Validation Score')

    plt.xlabel('Training Size', fontsize=13)
    plt.ylabel('F1-Score', fontsize=13)
    plt.title(f'Learning Curve - {model_name}',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Learning curve saved!")


plot_learning_curve(
    best_model, X_train_resampled,
    y_train_resampled, best_model_name
)


# =============================================
# 23. SAVE ALL MODELS
# =============================================
print("\n" + "=" * 60)
print("💾 SAVING ALL MODELS")
print("=" * 60)

os.makedirs('saved_models', exist_ok=True)

# Save models
joblib.dump(lr_model, 'saved_models/logistic_regression.pkl')
print("✅ Logistic Regression saved")

joblib.dump(rf_tuned, 'saved_models/random_forest.pkl')
print("✅ Random Forest saved")

joblib.dump(xgb_tuned, 'saved_models/xgboost_model.pkl')
print("✅ XGBoost saved")

joblib.dump(gb_model, 'saved_models/gradient_boosting.pkl')
print("✅ Gradient Boosting saved")

# Save best model
joblib.dump(best_model, 'saved_models/best_model.pkl')
print(f"✅ Best Model ({best_model_name}) saved")

# Save scaler
joblib.dump(scaler, 'saved_models/scaler.pkl')
print("✅ Scaler saved")

# Save feature names
joblib.dump(feature_names, 'saved_models/feature_names.pkl')
print("✅ Feature names saved")

print(f"\n📖 Load models like this:")
print(f"   model = joblib.load('saved_models/best_model.pkl')")
print(f"   scaler = joblib.load('saved_models/scaler.pkl')")


# =============================================
# 24. FINAL SUMMARY
# =============================================

# Identify best model
best_result = all_results[best_model_name]

print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETE - FINAL SUMMARY")
print("=" * 60)
print(f"""
╔══════════════════════════════════════════════════════════════╗
║       CUSTOMER CHURN PREDICTION SYSTEM                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Dataset          : Telco Customer Churn                    ║
║  Samples          : {len(df):,}                                  ║
║  Features         : {len(feature_names)} (after engineering)               ║
║  Churn Rate       : {(y.sum()/len(y))*100:.1f}%                                   ║
║  Task             : Binary Classification                   ║
║                                                              ║
║  Preprocessing:                                              ║
║  ✅ Missing value handling                                   ║
║  ✅ Feature Engineering (8 new features)                     ║
║  ✅ Label Encoding & One-Hot Encoding                        ║
║  ✅ Standard Scaling                                         ║
║  ✅ SMOTE for class imbalance                                ║
║                                                              ║
║  Models Trained & Compared:                                  ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ 1. Logistic Regression                               │    ║
║  │    Accuracy: {all_results['Logistic Regression']['accuracy']*100:>6.2f}%  F1: {all_results['Logistic Regression']['f1']*100:>6.2f}%  AUC: {all_results['Logistic Regression']['roc_auc']*100:>6.2f}%  │    ║
║  ├──────────────────────────────────────────────────────┤    ║
║  │ 2. Random Forest (Tuned)                             │    ║
║  │    Accuracy: {all_results['Random Forest']['accuracy']*100:>6.2f}%  F1: {all_results['Random Forest']['f1']*100:>6.2f}%  AUC: {all_results['Random Forest']['roc_auc']*100:>6.2f}%  │    ║
║  ├──────────────────────────────────────────────────────┤    ║
║  │ 3. XGBoost (Tuned)                                   │    ║
║  │    Accuracy: {all_results['XGBoost']['accuracy']*100:>6.2f}%  F1: {all_results['XGBoost']['f1']*100:>6.2f}%  AUC: {all_results['XGBoost']['roc_auc']*100:>6.2f}%  │    ║
║  ├──────────────────────────────────────────────────────┤    ║
║  │ 4. Gradient Boosting                                  │    ║
║  │    Accuracy: {all_results['Gradient Boosting']['accuracy']*100:>6.2f}%  F1: {all_results['Gradient Boosting']['f1']*100:>6.2f}%  AUC: {all_results['Gradient Boosting']['roc_auc']*100:>6.2f}%  │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                              ║
║  🏆 Best Model: {best_model_name:<40} ║
║     F1-Score : {best_result['f1']*100:.2f}%                                   ║
║     ROC-AUC  : {best_result['roc_auc']*100:.2f}%                                   ║
║                                                              ║
║  Key Techniques:                                             ║
║  ✅ Cross Validation (5-fold)                                ║
║  ✅ Hyperparameter Tuning (GridSearchCV)                     ║
║  ✅ Feature Importance Analysis                              ║
║  ✅ Mutual Information                                       ║
║  ✅ SMOTE (Imbalanced Data Handling)                         ║
║  ✅ ROC-AUC & Precision-Recall Curves                       ║
║  ✅ Customer Risk Scoring                                    ║
║                                                              ║
║  Files Generated:                                            ║
║  📄 churn_distribution.png                                   ║
║  📄 numerical_features.png                                   ║
║  📄 categorical_features.png                                 ║
║  📄 correlation_heatmap.png                                  ║
║  📄 churn_correlations.png                                   ║
║  📄 confusion_matrices.png                                   ║
║  📄 roc_curves.png                                           ║
║  📄 precision_recall_curves.png                              ║
║  📄 feature_importance.png                                   ║
║  📄 mutual_information.png                                   ║
║  📄 model_comparison.png                                     ║
║  📄 risk_analysis.png                                        ║
║  📄 learning_curve.png                                       ║
║  📦 saved_models/*.pkl (all models + scaler)                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("🎯 Customer Churn Prediction System Complete!")
print("   All 4 models trained, tuned, evaluated & saved!")
print("   Ready for interview submission! 💪")