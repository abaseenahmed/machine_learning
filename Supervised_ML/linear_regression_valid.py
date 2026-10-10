# ============================================================
# LINEAR REGRESSION VALIDATION - SCIKIT-LEARN
# Student: YOU. Mentor: Me. No copy-paste without understanding.
# ============================================================

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ---------- STEP 1: LOAD DATA ----------
data = fetch_california_housing()
X = data.data
y = data.target

print(f"X shape: {X.shape}")   # Expect (20640, 8)
print(f"y shape: {y.shape}")   # Expect (20640,)

# ---------- STEP 2: TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- STEP 3: PIPELINE (BEST PRACTICE) ----------
# A Pipeline chains transformer + estimator.
# WHY? Prevents data leakage. Automates fit/transform during CV.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# ---------- STEP 4: HOLD-OUT VALIDATION ----------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"\n[Hold-Out] MSE: {mse:.4f} | R2: {r2:.4f}")

# ---------- STEP 5: K-FOLD CROSS-VALIDATION ----------
# KFold splits data into K chunks. Model trains K times.
# Each iteration: (K-1) folds train, 1 fold validates.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
print(f"\n[5-Fold CV] R2 scores per fold: {cv_r2}")
print(f"[5-Fold CV] Mean R2: {cv_r2.mean():.4f} | Std: {cv_r2.std():.4f}")

# ---------- STEP 6: REGULARIZED MODELS ----------
# Ridge (L2 penalty) and Lasso (L1 penalty) fight overfitting.

for name, model in [
    ('Ridge (alpha=1.0)', Ridge(alpha=1.0)),
    ('Lasso (alpha=0.1)', Lasso(alpha=0.1))
]:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
    print(f"\n[{name}] Mean R2: {scores.mean():.4f} | Std: {scores.std():.4f}")