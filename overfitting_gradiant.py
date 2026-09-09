import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Creating the dataset
np.random.seed(42)

X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + np.random.normal(0, 2, 100)


# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Creating a high-degree polynomial model
model = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)


# Training the model
model.fit(X_train, y_train)


# Making predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


# Calculating MSE
training_mse = mean_squared_error(y_train, train_pred)
testing_mse = mean_squared_error(y_test, test_pred)


# Printing results
print(f"Training MSE: {training_mse:.4f}")
print(f"Testing MSE: {testing_mse:.4f}")


# Visualization
X_plot = np.linspace(-5, 5, 500).reshape(-1, 1)
y_plot_pred = model.predict(X_plot)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    label="Testing Data"
)

plt.plot(
    X_plot,
    y_plot_pred,
    label="Polynomial Regression"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Overfitting Example")
plt.legend()
plt.show()