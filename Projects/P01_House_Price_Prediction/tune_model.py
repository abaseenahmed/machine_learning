
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("data/cleaned_data.csv")

X = df.drop(columns=["price", "date"])
y = df["price"]

categorical_features = ["city", "statezip"]
numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 2. Keep the test set aside
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

# 4. Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=1
    ))
])

# 5. Candidate hyperparameters
param_grid = {
    "model__n_estimators": [150, 300],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.7, 1.0]
}

# 6. Cross-validation on log prices
cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, np.log1p(y_train))

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation MAE on log-price scale:")
print(f"{-search.best_score_:.4f}")

# 7. Evaluate the selected model once on the held-out test set
best_model = search.best_estimator_

predictions = np.expm1(best_model.predict(X_test))
predictions = np.maximum(predictions, 0)

print("\nHeld-out test results:")
print(f"MAE:  ${mean_absolute_error(y_test, predictions):,.2f}")
print(
    f"RMSE: ${mean_squared_error(y_test, predictions) ** 0.5:,.2f}"
)
print(f"R²:    {r2_score(y_test, predictions):.4f}")
