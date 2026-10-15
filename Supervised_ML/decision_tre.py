from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_modal = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_modal.fit(X_test, y_test)
y_pred = tree_modal.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy of model is: {accuracy}')