"""Beginner-friendly logistic regression example.

This script predicts whether a student will pass based on:
- hours studied
- attendance
- previous score

How the logic works:
1. We create sample training data.
2. We separate the input features (X) from the target (y).
3. We split the data into training and testing sets.
4. We train a logistic regression model on the training data.
5. We test the model on unseen data.
6. We interpret the prediction probability and model coefficients.

Logistic regression is used for binary classification problems, where the
output is one of two classes, such as pass/fail, yes/no, or 0/1.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split


def create_dataset():
    """Create the student dataset used in this example."""
    data = {
        "hours_studied": [
            1, 2, 2, 3, 3,
            4, 4, 5, 5, 6,
            6, 7, 7, 8, 8,
            9, 9, 10, 10, 11,
        ],
        "attendance": [
            50, 55, 58, 60, 62,
            65, 68, 70, 72, 74,
            76, 78, 80, 82, 84,
            86, 88, 90, 92, 95,
        ],
        "previous_score": [
            40, 42, 45, 47, 50,
            52, 55, 58, 60, 63,
            65, 68, 70, 73, 75,
            78, 80, 83, 86, 90,
        ],
        "passed": [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ],
    }
    return pd.DataFrame(data)


def prepare_features_and_target(df):
    """Pick the feature columns and the target column."""
    feature_columns = ["hours_studied", "attendance", "previous_score"]
    X = df[feature_columns]
    y = df["passed"]
    return X, y


def split_data(X, y):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Create and train the logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def print_section(title):
    """Simple helper to format output sections."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def interpret_coefficients(model, feature_names):
    """Show model coefficients and explain their meaning."""
    print("Intercept:", model.intercept_[0])
    print("\nA positive coefficient means the feature increases the chance of passing.")
    print("A negative coefficient means the feature decreases the chance of passing.\n")

    for feature, coefficient in zip(feature_names, model.coef_[0]):
        print(f"{feature}: {coefficient:.6f}")

    odds_ratios = np.exp(model.coef_[0])
    print("\nOdds ratios:")
    for feature, odds_ratio in zip(feature_names, odds_ratios):
        print(f"{feature}: {odds_ratio:.6f}")


def evaluate_model(model, X_test, y_test):
    """Return predictions and useful metrics for evaluation."""
    y_pred = model.predict(X_test)
    pass_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    loss = log_loss(y_test, pass_probability)

    return y_pred, pass_probability, accuracy, cm, report, loss


def predict_new_student(model):
    """Use the trained model to predict a new student."""
    new_student = pd.DataFrame({
        "hours_studied": [7],
        "attendance": [82],
        "previous_score": [75],
    })

    probability_of_passing = model.predict_proba(new_student)[0, 1]
    class_prediction = model.predict(new_student)[0]

    print("Student data:")
    print(new_student)
    print(f"\nProbability of passing: {probability_of_passing:.4f}")
    print(f"Probability of passing: {probability_of_passing * 100:.2f}%")

    if class_prediction == 1:
        print("Prediction: PASS")
    else:
        print("Prediction: FAIL")


def main():
    """Main flow of the program."""
    print("=" * 60)
    print("LOGISTIC REGRESSION EXAMPLE")
    print("Student Pass / Fail Prediction")
    print("=" * 60)
    print("\nIn simple words:")
    print("- We are teaching the model from past student examples.")
    print("- The model learns which patterns lead to passing.")
    print("- Then it predicts whether a new student is likely to pass.")

    # Step 1: Prepare dataset
    df = create_dataset()
    print_section("STEP 1: DATASET")
    print(df)

    # Step 2: Separate features and target
    X, y = prepare_features_and_target(df)

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    print_section("STEP 2: TRAIN / TEST SPLIT")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Make predictions on test data
    y_pred, pass_probability, accuracy, cm, report, loss = evaluate_model(model, X_test, y_test)
    print_section("STEP 3: PREDICTIONS")
    print("Actual labels:")
    print(y_test.values)
    print("\nPredicted labels:")
    print(y_pred)

    print_section("STEP 4: PROBABILITY OF PASSING")
    print(model.predict_proba(X_test))
    print("\nProbability of passing for each test student:")
    print(pass_probability)

    print_section("STEP 5: MODEL COEFFICIENTS")
    interpret_coefficients(model, X.columns)

    print_section("STEP 6: MODEL PERFORMANCE")
    print("Accuracy:", accuracy)
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)
    print("\nLog loss:", loss)

    print_section("STEP 7: ACTUAL VS PREDICTED")
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Pass_Probability": pass_probability,
    })
    print(results.to_string(index=False))

    print_section("STEP 8: PREDICT FOR A NEW STUDENT")
    predict_new_student(model)


if __name__ == "__main__":
    main()
