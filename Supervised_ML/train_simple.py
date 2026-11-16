"""Train a simple classifier using the local classification dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Datasets" / "classification.csv"
MODEL_PATH = PROJECT_DIR / "models" / "simple_classifier.joblib"
TARGET_COLUMN = "target"


def main() -> None:
    """Load data, train a logistic-regression pipeline, and save the model."""
    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")

    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    numeric_columns = features.select_dtypes(include="number").columns
    categorical_columns = features.select_dtypes(exclude="number").columns

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
