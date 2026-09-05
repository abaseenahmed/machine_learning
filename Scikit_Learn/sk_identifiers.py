# Import everything
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Data
iris = load_iris()
X = iris.data        # Shape: (150, 4)
y = iris.target      # Shape: (150,)

# 2. Train/Test Split (Holdout set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRANSFORMER: StandardScaler
scaler = StandardScaler()
# Learn the mean/std from TRAINING data ONLY
scaler.fit(X_train)   
# Transform BOTH sets using the TRAINING parameters
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. PREDICTOR: Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Check the learned parameters (note the trailing underscore)
print(f"Coefficients shape: {model.coef_.shape}")
print(f"Intercept: {model.intercept_}")