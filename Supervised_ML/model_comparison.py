"""Compare supervised classifiers on scikit-learn's breast-cancer dataset."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
MALIGNANT_LABEL = 0


def build_models() -> dict[str, Pipeline]:
    """Return classifiers with preprocessing contained in each CV pipeline."""
    classifiers = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Gaussian Naive Bayes": GaussianNB(),
    }
    return {
        name: Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }


def evaluate_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate held-out metrics, treating malignant cases as the positive class."""
    predictions = model.predict(X_test)
    class_labels = list(model.classes_)
    malignant_index = class_labels.index(MALIGNANT_LABEL)
    malignant_probabilities = model.predict_proba(X_test)[:, malignant_index]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "malignant_precision": float(
            precision_score(y_test, predictions, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "malignant_recall": float(
            recall_score(y_test, predictions, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "malignant_f1": float(
            f1_score(y_test, predictions, pos_label=MALIGNANT_LABEL, zero_division=0)
        ),
        "specificity": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test == MALIGNANT_LABEL, malignant_probabilities)
        ),
    }


def main() -> None:
    """Compare models, select by cross-validation ROC-AUC, and save artifacts."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "malignant_recall": make_scorer(recall_score, pos_label=MALIGNANT_LABEL),
        "malignant_f1": make_scorer(f1_score, pos_label=MALIGNANT_LABEL),
    }

    comparison_rows = []
    fitted_models = {}
    for name, model in build_models().items():
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        model.fit(X_train, y_train)
        fitted_models[name] = model
        row = {
            "model": name,
            "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
            "cv_roc_auc_std": cv_results["test_roc_auc"].std(),
            "cv_malignant_recall_mean": cv_results["test_malignant_recall"].mean(),
            "cv_malignant_f1_mean": cv_results["test_malignant_f1"].mean(),
            **{
                f"test_{metric}": score
                for metric, score in evaluate_holdout(model, X_test, y_test).items()
            },
        }
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_roc_auc_mean", "cv_malignant_recall_mean"],
        ascending=False,
        kind="mergesort",
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected_model = fitted_models[selected_name]
    selected_metrics = evaluate_holdout(selected_model, X_test, y_test)

    project_dir = Path(__file__).resolve().parent
    models_dir = project_dir / "models"
    reports_dir = project_dir / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    joblib.dump(selected_model, models_dir / "selected_model.joblib")
    summary = {
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "dataset_source": "scikit-learn load_breast_cancer (UCI Machine Learning Repository)",
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "classes": {
            str(label): str(name)
            for label, name in enumerate(dataset.target_names)
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "cross_validation": "5-fold stratified, shuffled; training set only",
        },
        "selection_metric": "mean cross-validation ROC-AUC",
        "selected_model": selected_name,
        "selected_model_cv_roc_auc_mean": float(comparison.iloc[0]["cv_roc_auc_mean"]),
        "selected_model_test_metrics": selected_metrics,
        "note": (
            "The holdout metrics are for this split only and do not establish "
            "clinical suitability."
        ),
    }
    with (reports_dir / "evaluation.json").open("w", encoding="utf-8") as report_file:
        json.dump(summary, report_file, indent=2)

    print(f"Dataset: {summary['dataset']} ({len(X)} samples, {X.shape[1]} features)")
    print("\nModel comparison (ranked by training CV ROC-AUC):")
    print(
        comparison[
            [
                "model",
                "cv_roc_auc_mean",
                "cv_malignant_recall_mean",
                "test_accuracy",
                "test_malignant_recall",
                "test_roc_auc",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nSelected model: {selected_name}")
    print(f"Saved model: {models_dir / 'selected_model.joblib'}")
    print(f"Saved reports: {reports_dir}")


if __name__ == "__main__":
    main()
