
# 1. Import libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 2. Load the dataset
wine = load_wine()

X = wine.data
y = wine.target

print("Dataset shape:", X.shape)
print("Feature names:", wine.feature_names)
print("Class names:", wine.target_names)

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create the SVM model
model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Predict a new sample
new_sample = wine.data[[0]]

# Scale the sample using the same scaler
new_sample_scaled = scaler.transform(new_sample)

prediction = model.predict(new_sample_scaled)

print("\nPredicted class:")
print(wine.target_names[prediction[0]])
