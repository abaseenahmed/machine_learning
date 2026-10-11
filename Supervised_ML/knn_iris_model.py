# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# STEP 2 + 3: Load data
iris = load_iris(as_frame=True)          # ✅ keeps iris.target_names available
iris_df = iris.frame                     # DataFrame with features + target
print(iris_df.head())

# STEP 4: Separate X and y
X = iris_df.drop('target', axis=1).values
y = iris_df['target'].values

# STEP 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 6: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# STEP 7: Train KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# STEP 8: Evaluate
y_pred = knn_model.predict(X_test_scaled)
print(f'\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))   # ✅ works now

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names, cmap='Blues'
)
plt.show()

# STEP 9: Find best K
print("\nFinding Best K...")
best_k, best_acc = 1, 0
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    if acc > best_acc:
        best_acc, best_k = acc, k
    print(f"K={k:2d} -> Accuracy: {acc * 100:.2f}%")
print(f"\n✅ Best K = {best_k} with Accuracy = {best_acc * 100:.2f}%")

# STEP 10: Predict new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_scaled = scaler.transform(new_flower)
pred_class = knn_model.predict(new_scaled)[0]
print(f"\n🎯 New flower predicted as: {iris.target_names[pred_class]}")

# STEP 11: Plot K vs Accuracy
k_values, accuracies = [], []
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test_scaled)))
    k_values.append(k)

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', color='blue')
plt.title('K vs Accuracy')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Accuracy')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()