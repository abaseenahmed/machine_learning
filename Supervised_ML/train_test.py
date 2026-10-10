import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Creating the dataset
np.random.seed(42)

X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + np.random.normal(0, 2, 100)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

print("CV Scores:", scores)
print("Mean CV MSE:", -scores.mean())