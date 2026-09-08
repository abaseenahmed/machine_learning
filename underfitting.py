import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Making the Error zeror and reducing undefitting problem
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# --------------------------------
# 1. Create nonlinear dataset
# --------------------------------
np.random.seed(42)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + np.random.normal(0, 2, 100)

# --------------------------------
# 2. Train/Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# 3. Create a Linear Regression model
# --------------------------------
model = LinearRegression()

# --------------------------------
# 4. Train the model
# --------------------------------
model.fit(X_train, y_train)

# --------------------------------
# 5. Predictions
# --------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# --------------------------------
# 6. Calculate errors
# --------------------------------
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)

# --------------------------------
# 7. Visualize
# --------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual Data")
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

# Reducing the underfitting problem.
model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print('After Reduced Underfitting Problem')
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)