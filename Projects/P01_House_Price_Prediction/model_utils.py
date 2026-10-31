from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "data" / "cleaned_data.csv"

NUMERIC_FEATURES = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
)
CATEGORICAL_FEATURES = ("city", "statezip")
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, "price")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load cleaned sales data and check the columns needed by the model."""
    data = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.loc[data["price"].gt(0)].dropna(subset=["price"])
    if data.empty:
        raise ValueError("Dataset contains no rows with a valid positive sale price.")
    if data.loc[:, FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing property feature values.")
    return data


def _build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Evaluate on a holdout split, then fit the deployment model on all rows."""
    features = data.loc[:, FEATURE_COLUMNS]
    target = data["price"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    evaluation_model = _build_pipeline()
    evaluation_model.fit(x_train, np.log1p(y_train))
    predictions = np.maximum(np.expm1(evaluation_model.predict(x_test)), 0)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }
    holdout_predictions = pd.DataFrame(
        {"actual": y_test.to_numpy(), "predicted": predictions}
    )

    final_model = _build_pipeline()
    final_model.fit(features, np.log1p(target))
    preprocessor = final_model.named_steps["preprocessor"]
    model: Any = final_model.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return final_model, metrics, holdout_predictions, importance


def predict_prices(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Convert the model's log-price output back to non-negative sale prices."""
    return np.maximum(np.expm1(model.predict(features)), 0)
