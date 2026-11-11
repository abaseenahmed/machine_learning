from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "cleaned_data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "churn_pipeline.joblib"
REPORT_DIR = PROJECT_DIR / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def align_features(
    features: pd.DataFrame, expected_columns: list[str]
) -> pd.DataFrame:
    if not expected_columns or len(expected_columns) != len(set(expected_columns)):
        raise ValueError(
            "The model must define a non-empty, unique feature_columns list."
        )

    missing = sorted(set(expected_columns) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(expected_columns))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError(
            "Evaluation data does not match the model schema ("
            + "; ".join(details)
            + ")."
        )

    return features.loc[:, expected_columns]


def load_evaluation_data(
    data_path: Path, expected_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(data_path)
    if "Churn" not in data:
        raise ValueError(f"{data_path} must contain a 'Churn' target column.")

    labels = data["Churn"].map({"Yes": 1, "No": 0})
    if labels.isna().any():
        invalid = sorted(data.loc[labels.isna(), "Churn"].astype(str).unique())
        raise ValueError(f"Unexpected Churn values: {invalid}")

    features = data.drop(columns=["Churn", "customerID"], errors="ignore")
    features = align_features(features, expected_columns)
    return features, labels.astype(int)


def evaluate(data_path: Path, model_path: Path) -> None:
    import joblib

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train.py first."
        )

    bundle: dict[str, Any] = joblib.load(model_path)
    feature_columns = bundle.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ValueError("Trained model bundle must contain a feature_columns list.")

    features, labels = load_evaluation_data(data_path, feature_columns)
    _, X_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "model": bundle["model_name"],
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["No churn (0)", "Churn (1)"],
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
        "test_rows": int(len(y_test)),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Selected model: {bundle['model_name']}")
    print(f"Decision threshold: {threshold:.3f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print("\nConfusion matrix (rows=actual, columns=predicted; No, Yes):")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            zero_division=0,
        )
    )

    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])
    print(
        f"False positives: {false_positives} loyal customers flagged; "
        f"false negatives: {false_negatives} churners missed."
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {metrics['roc_auc']:.3f}",
        color="#0b7a75",
        linewidth=2.5,
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Receiver operating characteristic",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "roc_curve.png", dpi=160)
    plt.close(figure)

    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_table.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print("\nTop permutation importances (ROC-AUC decrease when shuffled):")
    print(importance_table.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nEvaluation artifacts saved in {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained churn model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    evaluate(args.data, args.model)
