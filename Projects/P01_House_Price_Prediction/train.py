
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# 2. Separate features and target
X = df.drop(columns=["price", "date"])
y = df["price"]

# 3. Identify numerical and categorical features
categorical_features = ["city", "statezip"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 4. Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 5. Build preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 6. Define candidate models
models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=10.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# 7. Compare original-target and log-target training
results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    for target_type in ["Original Price", "Log Price"]:
        # Train on original prices or logarithmic prices
        if target_type == "Original Price":
            y_fit = y_train
        else:
            y_fit = np.log1p(y_train)

        pipeline.fit(X_train, y_fit)

        predictions = pipeline.predict(X_test)

        # Convert log predictions back to dollars
        if target_type == "Log Price":
            predictions = np.expm1(predictions)

        # House prices cannot be negative
        predictions = np.maximum(predictions, 0)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "Target": target_type,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"\n{name} | {target_type}")
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:    {r2:.4f}")

# 8. Compare all experiments
results_df = pd.DataFrame(results).sort_values("MAE")

print("\n\nMODEL COMPARISON")
print(results_df.to_string(index=False))
