from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "cleaned_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MINIMUM_RECALL = 0.75


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(data_path)
    if "Churn" not in data:
        raise ValueError(f"{data_path} must contain a 'Churn' target column.")

    labels = data["Churn"].map({"Yes": 1, "No": 0})
    if labels.isna().any():
        invalid = sorted(data.loc[labels.isna(), "Churn"].astype(str).unique())
        raise ValueError(f"Unexpected Churn values: {invalid}")

    features = data.drop(columns=["Churn", "customerID"], errors="ignore")
    if features.empty:
        raise ValueError("No predictor features remain after removing target and ID.")
    return features, labels.astype(int)


def make_preprocessor(
    numerical_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numerical", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, Any]:
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def metric_row(
    model: Pipeline, features: pd.DataFrame, labels: pd.Series, threshold: float = 0.5
) -> dict[str, float | int]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_precision": float(precision_score(labels, predictions, zero_division=0)),
        "test_recall": float(recall_score(labels, predictions, zero_division=0)),
        "test_f1": float(f1_score(labels, predictions, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(labels, probabilities)),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
    }


def choose_threshold(
    labels: pd.Series, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    choices: list[tuple[tuple[float, float, float], float, dict[str, float]]] = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        recall = float(recall_score(labels, predictions, zero_division=0))
        precision = float(precision_score(labels, predictions, zero_division=0))
        f1 = float(f1_score(labels, predictions, zero_division=0))
        if recall >= MINIMUM_RECALL:
            choices.append(((f1, precision, recall), float(threshold), {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }))

    if choices:
        _, threshold, metrics = max(choices, key=lambda item: item[0])
        return threshold, metrics

    # This fallback is only used if no threshold meets the recall constraint.
    best_threshold = float(candidates[0])
    predictions = (probabilities >= best_threshold).astype(int)
    return best_threshold, {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def parameter_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
        "Decision Tree": {
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 5, 10],
        },
        "Random Forest": {
            "model__n_estimators": [200],
            "model__max_depth": [None, 16],
            "model__min_samples_leaf": [1, 3],
        },
        "SVM": {"model__C": [0.5, 1.0, 5.0], "model__kernel": ["rbf"]},
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def train(data_path: Path) -> None:
    import joblib

    features, labels = load_data(data_path)
    numerical_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numerical_features
    ]
    preprocessor = make_preprocessor(numerical_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}
    for name, estimator in make_models().items():
        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", estimator)]
        )
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        pipeline.fit(X_train, y_train)
        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            scores = cv_scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(scores.mean())
            row[f"cv_{metric}_std"] = float(scores.std())
        row.update(metric_row(pipeline, X_test, y_test))
        results.append(row)
        pipelines[name] = pipeline

    comparison = pd.DataFrame(results).sort_values(
        ["cv_f1_mean", "cv_recall_mean", "cv_roc_auc_mean"],
        ascending=False,
    )
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nCross-validation and held-out comparison (default threshold = 0.50):")
    print(
        comparison[
            [
                "model",
                "cv_accuracy_mean",
                "cv_precision_mean",
                "cv_recall_mean",
                "cv_f1_mean",
                "cv_roc_auc_mean",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "test_fp",
                "test_fn",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    tune_names = (
        comparison.loc[comparison["model"] != "Baseline (most frequent)", "model"]
        .head(2)
        .tolist()
    )
    searches: list[tuple[str, GridSearchCV]] = []
    for name in tune_names:
        search = GridSearchCV(
            pipelines[name],
            parameter_grids()[name],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches.append((name, search))

    tuned_name, tuned_search = max(
        searches, key=lambda item: (item[1].best_score_, item[0])
    )
    final_pipeline: Pipeline = tuned_search.best_estimator_
    oof_probabilities = cross_val_predict(
        final_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, threshold_cv_metrics = choose_threshold(y_train, oof_probabilities)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final_pipeline.fit(X_train, y_train)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "model_name": tuned_name,
            "threshold": threshold,
            "feature_columns": features.columns.tolist(),
        },
        MODEL_DIR / "churn_pipeline.joblib",
    )

    search_summary = [
        {
            "model": name,
            "best_cv_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
        }
        for name, search in searches
    ]
    (REPORT_DIR / "training_summary.json").write_text(
        json.dumps(
            json_ready(
                {
                    "selected_model": tuned_name,
                    "selected_model_cv_f1": tuned_search.best_score_,
                    "best_parameters": tuned_search.best_params_,
                    "threshold": threshold,
                    "threshold_selection_minimum_recall": MINIMUM_RECALL,
                    "threshold_oof_metrics": threshold_cv_metrics,
                    "tuned_candidates": search_summary,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "target_mapping": {"No": 0, "Yes": 1},
                    "dropped_columns": ["customerID"],
                    "numerical_features": numerical_features,
                    "categorical_features": categorical_features,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    reference_values: dict[str, Any] = {}
    widget_metadata: dict[str, Any] = {}
    for column in numerical_features:
        series = X_train[column]
        reference_values[column] = float(series.median())
        widget_metadata[column] = {
            "kind": "numeric",
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "median": float(series.median()),
            "integer": bool(pd.api.types.is_integer_dtype(series.dtype)),
            "options": sorted(series.dropna().unique().tolist())
            if series.nunique() <= 8
            else [],
        }
    for column in categorical_features:
        series = X_train[column]
        reference_values[column] = str(series.mode().iloc[0])
        widget_metadata[column] = {
            "kind": "categorical",
            "options": sorted(series.dropna().astype(str).unique().tolist()),
            "default": str(series.mode().iloc[0]),
        }
    (MODEL_DIR / "feature_metadata.json").write_text(
        json.dumps(
            json_ready(
                {
                    "features": widget_metadata,
                    "reference_values": reference_values,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSelected {tuned_name} (best tuning CV F1={tuned_search.best_score_:.3f}); "
        f"OOF threshold={threshold:.3f}, OOF precision={threshold_cv_metrics['precision']:.3f}, "
        f"recall={threshold_cv_metrics['recall']:.3f}, F1={threshold_cv_metrics['f1']:.3f}."
    )
    print(f"Saved pipeline and metadata in {MODEL_DIR}")
    print(f"Saved comparison and training summary in {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train(args.data)
