# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# 2. CREATE DATASET
# =====================================================

data = {
    "hours_studied": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11
    ],

    "attendance": [
        50, 55, 58, 60, 62, 65, 68, 70, 72, 74,
        76, 78, 80, 82, 84, 86, 88, 90, 92, 95
    ],

    "previous_score": [
        40, 42, 45, 47, 50, 52, 55, 58, 60, 63,
        65, 68, 70, 73, 75, 78, 80, 83, 86, 90
    ],

    "passed": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# =====================================================
# 3. SEPARATE FEATURES AND TARGET
# =====================================================

X = df[
    ["hours_studied", "attendance", "previous_score"]
]

y = df["passed"]


# =====================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================================
# 5. SCALE FEATURES
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =====================================================
# 6. CREATE KNN MODEL
# =====================================================

knn = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)


# =====================================================
# 7. TRAIN THE MODEL
# =====================================================

knn.fit(X_train_scaled, y_train)


# =====================================================
# 8. PREDICT TEST DATA
# =====================================================

y_pred = knn.predict(X_test_scaled)

print("\nActual classes:")
print(y_test.values)

print("\nPredicted classes:")
print(y_pred)


# =====================================================
# 9. EVALUATE THE MODEL
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# =====================================================
# 10. PREDICT A NEW STUDENT
# =====================================================

new_student = pd.DataFrame({
    "hours_studied": [7],
    "attendance": [82],
    "previous_score": [75]
})

# Use the SAME scaler fitted on training data
new_student_scaled = scaler.transform(new_student)

prediction = knn.predict(new_student_scaled)

probabilities = knn.predict_proba(new_student_scaled)

print("\nNew student:")
print(new_student)

print("\nPredicted class:", prediction[0])

print("Probabilities [Fail, Pass]:", probabilities[0])

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")