import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = {
    "age": [22, 25, 28, 30, 35, 40, 42, 45, 50, 55,
            60, 23, 27, 32, 38, 41, 48, 52, 57, 62],

    "experience": [1, 2, 4, 5, 8, 12, 15, 18, 22, 25,
                   30, 1, 3, 6, 10, 14, 19, 23, 27, 32],

    "education_years": [16, 16, 16, 18, 18, 18, 16, 18, 16, 14,
                         14, 16, 16, 18, 18, 16, 18, 16, 14, 14],

    "salary": [35000, 42000, 50000, 60000, 75000, 95000,
               110000, 130000, 155000, 170000,
               190000, 38000, 47000, 65000, 85000, 105000,
               125000, 145000, 165000, 200000]
}

df = pd.DataFrame(data)


# Features
X = df[
    [
        "age",
        "experience",
        "education_years"
    ]
]


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


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


print("MAE:", mae)
print("MSE:", mse)


# Coefficients
print("Coefficients:", model.coef_)

# Intercept
print("Intercept:", model.intercept_)
