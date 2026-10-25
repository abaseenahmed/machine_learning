"""Train and evaluate a Gaussian Naive Bayes classifier on scikit-learn's Wine dataset."""

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main() -> None:
    wine = load_wine()
    features = wine.data
    labels = wine.target

    print(f"Dataset shape: {features.shape}")
    print(f"Features: {wine.feature_names}")
    print(f"Classes: {wine.target_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=wine.target_names,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    sample = X_test[[0]]
    predicted_class = model.predict(sample)[0]
    actual_class = y_test[0]
    print(f"\nPredicted class: {wine.target_names[predicted_class]}")
    print(f"Actual class: {wine.target_names[actual_class]}")


if __name__ == "__main__":
    main()
