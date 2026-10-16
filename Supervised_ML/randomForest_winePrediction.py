from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score, f1_score)

wine = load_wine()
X = wine.data
y = wine.target

print('Some Details About the Dataset: ')
print("Feature names:", wine.feature_names)
print("Class names:", wine.target_names)
print("Dataset shape:", wine.data.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
cls_report = classification_report(y_test, y_pred, target_names=wine.target_names)
conf_mat = confusion_matrix(y_test, y_pred)

print(f'The Accuracy of Model is : \n{acc_score}')
print(f'The Classification Report of Model is : \n{cls_report}')
print(f'The Confusion Matrix of Model is : \n{conf_mat}')

precision = precision_score(
    y_test, y_pred, average="weighted"
)

recall = recall_score(
    y_test, y_pred, average="weighted"
)

f1 = f1_score(
    y_test, y_pred, average="weighted"
)

print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("Weighted F1-score:", f1)

new_sample = wine.data[[0]]
new_pred = model.predict(new_sample)

print("Predicted class:", wine.target_names[new_pred[0]])

