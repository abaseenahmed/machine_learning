import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [60, 62, 65, 68, 72, 75, 80, 85, 90, 95],
    "previous_score": [40, 45, 48, 52, 58, 62, 68, 73, 78, 84],
    "final_score": [42, 47, 53, 57, 64, 68, 75, 81, 86, 93]
}

df = pd.DataFrame(data)

X = df[[
    "hours_studied",
    "attendance",
    "previous_score"
]]

y = df["final_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


print("Elastic Net Regression")
print("---------------------")
print("MAE:", mae)
print("MSE:", mse)

print("Intercept:", model.intercept_)

print("Coefficients:", model.coef_)