import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. DATASET
# ============================================================

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


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df[[
    "hours_studied",
    "attendance",
    "previous_score"
]]

y = df["final_score"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)

print("Actual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)


# ============================================================
# 6. INDIVIDUAL ERROR ANALYSIS
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})


# Error
# Actual - Predicted

results["Error"] = (
    results["Actual"] -
    results["Predicted"]
)


# Absolute Error

results["Absolute_Error"] = (
    results["Error"].abs()
)


# Squared Error

results["Squared_Error"] = (
    results["Error"] ** 2
)


# Percentage Error

results["Percentage_Error"] = (
    results["Absolute_Error"] /
    results["Actual"]
) * 100


print("\n" + "=" * 60)
print("INDIVIDUAL ERROR ANALYSIS")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 7. MAE
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


# ============================================================
# 8. MSE
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)


# ============================================================
# 9. RMSE
# ============================================================

rmse = np.sqrt(mse)


# ============================================================
# 10. R-SQUARED
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 11. MAPE
# ============================================================

mape = np.mean(
    np.abs(
        (y_test - y_pred) /
        y_test
    )
) * 100


# ============================================================
# 12. ADJUSTED R-SQUARED
# ============================================================

n = len(y_test)
p = X_test.shape[1]

if n > p + 1:

    adjusted_r2 = (
        1 -
        (1 - r2) *
        (n - 1) /
        (n - p - 1)
    )

else:

    adjusted_r2 = np.nan


# ============================================================
# 13. MEAN ERROR / BIAS
# ============================================================

mean_error = results["Error"].mean()


# ============================================================
# 14. MAXIMUM ERROR
# ============================================================

maximum_error = results["Absolute_Error"].max()


# ============================================================
# 15. METRICS SUMMARY
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "MAPE",
        "R²",
        "Adjusted R²",
        "Mean Error",
        "Maximum Absolute Error"
    ],

    "Value": [
        mae,
        mse,
        rmse,
        mape,
        r2,
        adjusted_r2,
        mean_error,
        maximum_error
    ]
})


print("\n" + "=" * 60)
print("REGRESSION METRICS")
print("=" * 60)

print(
    metrics.to_string(
        index=False
    )
)


# ============================================================
# 16. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    f"MAE: The model's average absolute error is "
    f"{mae:.4f} points."
)

print(
    f"MSE: The average squared error is "
    f"{mse:.4f}."
)

print(
    f"RMSE: The model's error in the original "
    f"score unit is approximately {rmse:.4f} points."
)

print(
    f"MAPE: The model's average percentage error "
    f"is approximately {mape:.4f}%."
)

print(
    f"R²: The model explains approximately "
    f"{r2 * 100:.4f}% of the variation in the target "
    f"on this test set."
)

if not np.isnan(adjusted_r2):

    print(
        f"Adjusted R²: {adjusted_r2:.4f}"
    )

else:

    print(
        "Adjusted R²: Cannot be meaningfully calculated "
        "because there are too few test observations "
        "relative to the number of predictors."
    )


if mean_error > 0:

    print(
        "Bias: The model tends to underpredict."
    )

elif mean_error < 0:

    print(
        "Bias: The model tends to overpredict."
    )

else:

    print(
        "Bias: No average directional error."
    )


print(
    f"Largest absolute prediction error: "
    f"{maximum_error:.4f} points."
)


# ============================================================
# 17. FIND WORST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("WORST PREDICTIONS")
print("=" * 60)

worst_predictions = results.sort_values(
    "Absolute_Error",
    ascending=False
)

print(
    worst_predictions.to_string(
        index=False
    )
)