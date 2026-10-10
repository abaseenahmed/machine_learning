"""
KNN Classification - Simple Version
Dataset: clustering.csv (2 features, 4 classes)
"""

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# STEP 2: Load Data
df = pd.read_csv('Datasets/clustering.csv')
print(df.head())
print(f"Shape: {df.shape}")

# STEP 3: Separate Features and Target
X = df[['feature_1', 'feature_2']].values
y = df['cluster'].values

# STEP 4: Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 5: Scale Features (important for KNN!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# STEP 6: Create and Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# STEP 7: Predict and Check Accuracy
y_pred = knn.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# STEP 8: Try Different K Values
for k in [1, 3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"K={k} -> Accuracy: {acc * 100:.2f}%")

# STEP 9: Visualize Decision Boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis')
plt.title('KNN Decision Boundary (K=5)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# STEP 10: Predict a New Point
new_point = np.array([[0.5, -0.3]])
new_point = scaler.transform(new_point)
print(f"\nNew point belongs to class: {knn.predict(new_point)[0]}")