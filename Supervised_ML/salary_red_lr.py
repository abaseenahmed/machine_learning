import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = {
    "experience": [1, 2, 4, 5, 8, 12],
    "salary": [35000, 42000, 50000, 60000, 75000, 95000]
}

df = pd.DataFrame(data)


# Feature
X = df[["experience"]]

# Target
y = df["salary"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


print("MAE:", mae)
print("MSE:", mse)


# Model parameters
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)