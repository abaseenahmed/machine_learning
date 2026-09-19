import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Dataset
data = {
    "hours_studied": [
        2, 3, 4, 5, 6,
        7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 21
    ],

    "attendance": [
        60, 62, 65, 68, 70,
        72, 75, 77, 80, 82,
        84, 85, 87, 89, 90,
        91, 93, 94, 96, 98
    ],

    "previous_score": [
        45, 48, 50, 54, 57,
        60, 62, 65, 67, 70,
        72, 74, 76, 79, 81,
        83, 86, 88, 91, 94
    ],

    "final_score": [
        48, 51, 54, 58, 61,
        64, 67, 70, 73, 76,
        79, 81, 84, 87, 89,
        91, 94, 96, 98, 100
    ]
}

df = pd.DataFrame(data)

x = df[['hours_studied', 'attendance', 'previous_score']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Actual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_predict)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)