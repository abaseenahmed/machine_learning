import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def create_dataset() -> pd.DataFrame:
    """Create the sample student performance dataset."""
    data = {
        "hours_studied": [
            2, 3, 4, 5, 6,
            7, 8, 9, 10, 11,
            12, 13, 14, 15, 16,
            17, 18, 19, 20, 21,
        ],
        "attendance": [
            60, 62, 65, 68, 70,
            72, 75, 77, 80, 82,
            84, 85, 87, 89, 90,
            91, 93, 94, 96, 98,
        ],
        "previous_score": [
            45, 48, 50, 54, 57,
            60, 62, 65, 67, 70,
            72, 74, 76, 79, 81,
            83, 86, 88, 91, 94,
        ],
        "final_score": [
            48, 51, 54, 58, 61,
            64, 67, 70, 73, 76,
            79, 81, 84, 87, 89,
            91, 94, 96, 98, 100,
        ],
    }
    return pd.DataFrame(data)


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    absolute_error = np.abs(y_true.to_numpy() - y_pred)
    percentage_error = np.divide(
        absolute_error,
        y_true.to_numpy(),
        out=np.zeros_like(absolute_error, dtype=float),
        where=np.abs(y_true.to_numpy()) > 0,
    ) * 100
    mape = np.mean(percentage_error)

    n = len(y_true)
    p = len(y_true.to_frame().columns) if hasattr(y_true, "to_frame") else 1

    if n > p + 1:
        adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    else:
        adjusted_r2 = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }


def print_section(title: str) -> None:
    """Print a clean section header."""
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)


def evaluate_model() -> None:
    """Train a regression model and evaluate its performance."""
    df = create_dataset()

    features = ["hours_studied", "attendance", "previous_score"]
    target = "final_score"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print_section("DATA SPLIT")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluation_df = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": y_pred,
    })
    evaluation_df["Error"] = evaluation_df["Actual"] - evaluation_df["Predicted"]
    evaluation_df["Absolute_Error"] = evaluation_df["Error"].abs()
    evaluation_df["Squared_Error"] = evaluation_df["Error"] ** 2
    evaluation_df["Percentage_Error"] = (
        evaluation_df["Absolute_Error"] / evaluation_df["Actual"]
    ) * 100

    print_section("ACTUAL VS PREDICTED")
    print(evaluation_df[["Actual", "Predicted"]].to_string(index=False))

    print_section("INDIVIDUAL ERROR ANALYSIS")
    print(evaluation_df.to_string(index=False))

    metrics = compute_regression_metrics(y_test, y_pred)
    metrics_summary = pd.DataFrame(
        {
            "Metric": ["MAE", "MSE", "RMSE", "MAPE", "R²", "Adjusted R²"],
            "Value": [
                metrics["MAE"],
                metrics["MSE"],
                metrics["RMSE"],
                metrics["MAPE"],
                metrics["R2"],
                metrics["Adjusted_R2"],
            ],
        }
    )

    print_section("REGRESSION METRICS")
    print(metrics_summary.to_string(index=False))

    mean_error = evaluation_df["Error"].mean()
    maximum_error = evaluation_df["Absolute_Error"].max()

    print_section("INTERPRETATION")
    print(f"MAE: The model's average absolute error is {metrics['MAE']:.4f} points.")
    print(f"MSE: The average squared error is {metrics['MSE']:.4f}.")
    print(f"RMSE: The model's error in the original score unit is approximately {metrics['RMSE']:.4f} points.")
    print(f"MAPE: The model's average percentage error is approximately {metrics['MAPE']:.4f}%.")
    print(f"R²: The model explains approximately {metrics['R2'] * 100:.4f}% of the variation in the target on the test set.")

    if pd.notna(metrics["Adjusted_R2"]):
        print(f"Adjusted R²: {metrics['Adjusted_R2']:.4f}")
    else:
        print(
            "Adjusted R²: Cannot be meaningfully calculated because there are too few test observations "
            "relative to the number of predictors."
        )

    if mean_error > 0:
        print("Bias: The model tends to underpredict.")
    elif mean_error < 0:
        print("Bias: The model tends to overpredict.")
    else:
        print("Bias: No average directional error.")

    print(f"Largest absolute prediction error: {maximum_error:.4f} points.")

    print_section("WORST PREDICTIONS")
    worst_predictions = evaluation_df.sort_values("Absolute_Error", ascending=False)
    print(worst_predictions.to_string(index=False))


if __name__ == "__main__":
    evaluate_model()
