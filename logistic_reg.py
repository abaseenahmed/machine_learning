# ============================================================
# LOGISTIC REGRESSION
# Student Pass / Fail Prediction
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss
)


# ============================================================
# 1. DATASET
# ============================================================

data = {
    "hours_studied": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 8,
        9, 9, 10, 10, 11
    ],

    "attendance": [
        50, 55, 58, 60, 62,
        65, 68, 70, 72, 74,
        76, 78, 80, 82, 84,
        86, 88, 90, 92, 95
    ],

    "previous_score": [
        40, 42, 45, 47, 50,
        52, 55, 58, 60, 63,
        65, 68, 70, 73, 75,
        78, 80, 83, 86, 90
    ],

    "passed": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("DATASET")
print("=" * 60)

print(df)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df[
    [
        "hours_studied",
        "attendance",
        "previous_score"
    ]
]

y = df["passed"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression()

model.fit(X_train, y_train)


# ============================================================
# 5. PREDICT CLASS
# ============================================================

y_pred = model.predict(X_test)


print("\n" + "=" * 60)
print("CLASS PREDICTIONS")
print("=" * 60)

print("Actual:")
print(y_test.values)

print("\nPredicted:")
print(y_pred)


# ============================================================
# 6. PREDICT PROBABILITY
# ============================================================

y_probability = model.predict_proba(X_test)

print("\n" + "=" * 60)
print("PREDICTED PROBABILITIES")
print("=" * 60)

print(y_probability)


# Probability of class 1 = Passed

pass_probability = model.predict_proba(X_test)[:, 1]

print("\nProbability of Passing:")
print(pass_probability)


# ============================================================
# 7. COEFFICIENTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COEFFICIENTS")
print("=" * 60)

print("Intercept:", model.intercept_[0])

for feature, coefficient in zip(
    X.columns,
    model.coef_[0]
):
    print(
        f"{feature}: {coefficient:.6f}"
    )


# ============================================================
# 8. ODDS RATIOS
# ============================================================

odds_ratios = np.exp(
    model.coef_[0]
)

print("\n" + "=" * 60)
print("ODDS RATIOS")
print("=" * 60)

for feature, coefficient, odds_ratio in zip(
    X.columns,
    model.coef_[0],
    odds_ratios
):

    print(
        f"{feature}: "
        f"Coefficient = {coefficient:.6f}, "
        f"Odds Ratio = {odds_ratio:.6f}"
    )


# ============================================================
# 9. CREATE COEFFICIENT + ODDS RATIO TABLE
# ============================================================

coefficient_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0],
    "Odds_Ratio": np.exp(model.coef_[0])
})

print("\n" + "=" * 60)
print("COEFFICIENT / ODDS RATIO TABLE")
print("=" * 60)

print(
    coefficient_table.to_string(
        index=False
    )
)


# ============================================================
# 10. LOG LOSS
# ============================================================

logloss = log_loss(
    y_test,
    pass_probability
)

print("\n" + "=" * 60)
print("LOG LOSS")
print("=" * 60)

print("Log Loss:", logloss)


# ============================================================
# 11. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("ACCURACY")
print("=" * 60)

print("Accuracy:", accuracy)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 14. ACTUAL VS PREDICTED PROBABILITY
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Pass_Probability": pass_probability
})

print("\n" + "=" * 60)
print("PREDICTION ANALYSIS")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 15. PREDICT A NEW STUDENT
# ============================================================

new_student = pd.DataFrame({
    "hours_studied": [7],
    "attendance": [82],
    "previous_score": [75]
})


new_probability = model.predict_proba(
    new_student
)[0, 1]

new_prediction = model.predict(
    new_student
)[0]


print("\n" + "=" * 60)
print("NEW STUDENT PREDICTION")
print("=" * 60)

print("Student:")
print(new_student)

print(
    f"\nProbability of Passing: "
    f"{new_probability:.4f}"
)

print(
    f"Probability of Passing: "
    f"{new_probability * 100:.2f}%"
)

if new_prediction == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")