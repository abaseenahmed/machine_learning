from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
wine = load_wine()
X = wine.data
y = wine.target

print("Dataset shape:", X.shape)
print("Features:", wine.feature_names)
print("Classes:", wine.target_names)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Create the model
model = GaussianNB()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=wine.target_names,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Predict one held-out sample
sample = X_test[[0]]
prediction = model.predict(sample)

print("Predicted class:", wine.target_names[prediction[0]])
print("Actual class:", wine.target_names[y_test[0]])