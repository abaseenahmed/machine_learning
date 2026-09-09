import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Create dataset
np.random.seed(42)

X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + np.random.normal(0, 2, 100)


# First split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# Second split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)


print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Validation Model 1 Using LinearRegression()
model_1 = LinearRegression()
model_1.fit(X_train, y_train)
val_pred_1 = model_1.predict(X_val)
val_mse_1 = mean_squared_error(y_val, val_pred_1)
print("Linear Regression Validation MSE:", val_mse_1)

# Validation Model 2 Using PolynomialRegression of Degree 2
model_2 = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
model_2.fit(X_train, y_train)
val_pred_2 = model_2.predict(X_val)
val_mse_2 = mean_squared_error(y_val, val_pred_2)
print("Polynomial Degree 2 Validation MSE:", val_mse_2)

# Validation Model 2 Using PolynomialRegression of Degree 10
model_3 = make_pipeline(
    PolynomialFeatures(degree=10),
    LinearRegression()
)
model_3.fit(X_train, y_train)
val_pred_3 = model_3.predict(X_val)
val_mse_3 = mean_squared_error(y_val, val_pred_3)
print("Polynomial Degree 10 Validation MSE:", val_mse_3)

# Testing the final model from the above 3 models which has least MSE
final_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
final_model.fit(X_train, y_train)
test_pred = final_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_pred)
print("Final Test MSE:", test_mse)