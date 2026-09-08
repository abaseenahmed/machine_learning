import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + np.random.normal(0, 2, 100)

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Creating a LinearRegression() model.
model = LinearRegression()

# Training the model using .fit()
model.fit(X_train, y_train)

# Making predictions for taining and testing
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculating MSE (Mean Squared Error) For Training and Testing Predictions
training_mse = mean_squared_error(y_train, train_pred)
testing_mse = mean_squared_error(y_test, test_pred)

# Printing both MSE values
print(f'Traing MSE: {training_mse}')
print(f'Testing MSE: {testing_mse}')

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Actual Data')
plt.plot(
    X,
    model.predict(X),
    label="Linear Regression"
)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Underfitting Example")
plt.legend()
plt.show()