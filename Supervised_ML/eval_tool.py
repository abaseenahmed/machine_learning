
# 1. Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef
)

import matplotlib.pyplot as plt


# 2. Load the Iris dataset
iris = load_iris()

X = iris.data       # Flower measurements
y = iris.target     # Flower species

print("Features:", iris.feature_names)
print("Classes:", iris.target_names)
print("Dataset size:", X.shape)


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Scale the data (important for KNN)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


# 6. Make predictions
y_pred = model.predict(X_test)

# Predicted probabilities for each class
y_prob = model.predict_proba(X_test)


# 7. Calculate evaluation metrics
print("\n--- Evaluation Results ---")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Balanced Accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred, average="weighted"))

print("Recall:",
      recall_score(y_test, y_pred, average="weighted"))

print("F1 Score:",
      f1_score(y_test, y_pred, average="weighted"))

print("Log Loss:",
      log_loss(y_test, y_prob))

print("Matthews Correlation Coefficient:",
      matthews_corrcoef(y_test, y_pred))


# 8. Classification Report
print("\n--- Classification Report ---")

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("--- Confusion Matrix ---")
print(cm)


# 10. Display Confusion Matrix as a graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot(cmap="Blues")
plt.title("Iris Classification Confusion Matrix")
plt.show()
