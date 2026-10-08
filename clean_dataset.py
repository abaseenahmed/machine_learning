import pandas as pd
import numpy as np

# =========================================================
# 1. Load the dataset
# =========================================================
df = pd.read_csv('synthetic_ml_dataset.csv')

print("=" * 60)
print("STEP 1: INITIAL INSPECTION")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nBasic stats:\n{df.describe(include='all').T}")

# =========================================================
# 2. INSPECTION (run before cleaning to see problems)
# =========================================================
print("\n--- Numerical summary ---")
print(df.describe())

print("\n--- Categorical summary ---")
print(df.describe(include='object'))

print("\n--- Unique values in categorical columns ---")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique -> {df[col].unique()[:5]}")

# =========================================================
# 3. DROP IRRELEVANT COLUMNS
# =========================================================
# employee_id -> just an identifier, no predictive power
# name        -> free text, would need NLP to be useful
df_clean = df.drop(columns=['employee_id', 'name'])

print(f"After dropping ID/name: {df_clean.shape}")

# =========================================================
# 4. REMOVE DUPLICATES
# =========================================================
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
after = df_clean.shape[0]

print(f"Removed {before - after} duplicate rows. Remaining: {after}")

# =========================================================
# 5. HANDLE MISSING VALUES
# =========================================================
# Check them first
missing = df_clean.isnull().sum()
missing = missing[missing > 0]
print(f"\nColumns with missing values:\n{missing}")

# ---- Numeric columns: impute with median (robust to outliers)
num_cols_with_na = ['income', 'satisfaction']
for col in num_cols_with_na:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
    print(f"Filled '{col}' missing values with median = {median_val:.2f}")

# ---- Categorical columns: impute with mode
cat_cols_with_na = ['city']
for col in cat_cols_with_na:
    mode_val = df_clean[col].mode()[0]
    df_clean[col] = df_clean[col].fillna(mode_val)
    print(f"Filled '{col}' missing values with mode = '{mode_val}'")

# =========================================================
# 6. FIX DATA TYPES
# =========================================================
# Convert categorical text columns to pandas 'category' dtype
cat_cols = ['gender', 'city', 'department', 'education_level', 'remote_work']
for col in cat_cols:
    df_clean[col] = df_clean[col].astype('category')

# Ensure numeric columns are actually numeric (int/float)
num_cols = ['age', 'education_years', 'experience_years',
            'hours_worked', 'income', 'satisfaction', 'salary']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Target (classification) as category too
df_clean['high_earner'] = df_clean['high_earner'].astype('category')

print("\nDtypes after conversion:")
print(df_clean.dtypes)

# =========================================================
# 7. OUTLIER DETECTION (IQR method)
# =========================================================
def detect_outliers_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series[(series < lower) | (series > upper)]

for col in ['income', 'satisfaction', 'hours_worked', 'salary']:
    outliers = detect_outliers_iqr(df_clean[col])
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df_clean)*100:.2f}%)")

def cap_outliers(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower, upper)

for col in ['income', 'satisfaction', 'hours_worked', 'salary']:
    df_clean[col] = cap_outliers(df_clean[col])

print("\nOutliers capped using IQR rule.")

# =========================================================
# 8. SANITY CHECKS
# =========================================================
# Age should be between 18 and 100
print(f"Age range: {df_clean['age'].min()} - {df_clean['age'].max()}")

# Hours worked should be positive
assert (df_clean['hours_worked'] > 0).all(), "Negative hours found!"

# Satisfaction between 1 and 10
assert df_clean['satisfaction'].between(1, 10).all(), "Satisfaction out of range!"

# Income positive
assert (df_clean['income'] > 0).all(), "Negative income found!"

print("✅ All sanity checks passed.")

# =========================================================
# 9. SPLIT FEATURES AND TARGETS
# =========================================================
# Choose ONE target depending on your task:
#   - Classification  -> 'high_earner' (Yes/No)
#   - Regression      -> 'salary'       (continuous)

target_class = 'high_earner'
target_reg   = 'salary'

# Features = everything except the two targets
X = df_clean.drop(columns=[target_class, target_reg])
y_class = df_clean[target_class]
y_reg   = df_clean[target_reg]

print(f"X shape: {X.shape}")
print(f"y_class distribution:\n{y_class.value_counts()}")

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print(f"\nAfter one-hot encoding: {X_encoded.shape}")
print(X_encoded.head())

# =========================================================
# 10. SAVE CLEANED DATA
# =========================================================
# Save the cleaned (but still human-readable) version
df_clean.to_csv('cleaned_dataset.csv', index=False)

# Save one-hot encoded features + targets
X_encoded.to_csv('X_features_onehot.csv', index=False)
y_class.to_csv('y_classification.csv', index=False)
y_reg.to_csv('y_regression.csv', index=False)

print("\n✅ Saved:")
print("  cleaned_dataset.csv        -> cleaned, human-readable")
print("  X_features_onehot.csv      -> one-hot encoded features")
print("  y_classification.csv       -> classification target")
print("  y_regression.csv           -> regression target")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("FINAL CLEANED DATASET SUMMARY")
print("=" * 60)
print(f"Shape: {df_clean.shape}")
print(f"Missing values: {df_clean.isnull().sum().sum()}")
print(f"Duplicates: {df_clean.duplicated().sum()}")
print(f"Dtypes:\n{df_clean.dtypes}")