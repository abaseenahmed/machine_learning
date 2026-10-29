
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the cleaned data
df = pd.read_csv("data/cleaned_data.csv")

X = df.drop(columns=["price", "date"])
y = df["price"]

categorical_features = ["city", "statezip"]
numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 2. Recreate the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numerical_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 4. Train the best current candidate on log prices
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, np.log1p(y_train))

# 5. Predict in dollars
predictions = np.expm1(model.predict(X_test))
predictions = np.maximum(predictions, 0)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
print("R²:", round(r2_score(y_test, predictions), 4))


# 6. Compare actual and predicted prices
comparison = X_test.copy()

comparison["Actual Price"] = y_test
comparison["Predicted Price"] = predictions

comparison["Absolute Error"] = (
    comparison["Actual Price"] - comparison["Predicted Price"]
).abs()

comparison["Residual"] = (
    comparison["Actual Price"] - comparison["Predicted Price"]
)

# 7. Inspect the largest errors with property details
columns_to_show = [
    "Actual Price",
    "Predicted Price",
    "Absolute Error",
    "Residual",
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "city",
    "statezip"
]

print("\nLargest prediction errors:")

print(
    comparison.nlargest(10, "Absolute Error")[columns_to_show]
    .round(2)
    .to_string(index=False)
)

# 8. Evaluate performance by actual price range
comparison["Price Range"] = pd.cut(
    comparison["Actual Price"],
    bins=[0, 500_000, 1_000_000, 2_000_000, float("inf")],
    labels=[
        "Under $500K",
        "$500K-$1M",
        "$1M-$2M",
        "Over $2M"
    ]
)

print("\nPerformance by actual price range:")

for price_range, group in comparison.groupby(
    "Price Range", observed=True
):
    mae = group["Absolute Error"].mean()
    rmse = (
        group["Residual"].pow(2).mean()
    ) ** 0.5

    print(f"\n{price_range} | Houses: {len(group)}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")

# 9. Actual vs predicted graph
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=comparison,
    x="Actual Price",
    y="Predicted Price",
    alpha=0.6
)

max_price = max(
    comparison["Actual Price"].max(),
    comparison["Predicted Price"].max()
)

plt.plot([0, max_price], [0, max_price], "r--")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 10. Residual distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=comparison,
    x="Residual",
    bins=40,
    kde=True
)

plt.axvline(0, color="red", linestyle="--")
plt.title("Distribution of Prediction Residuals")
plt.tight_layout()
plt.show()
