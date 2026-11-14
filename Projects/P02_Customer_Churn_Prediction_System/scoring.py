from __future__ import annotations

import pandas as pd


def align_customer_features(
    customers: pd.DataFrame, expected_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series | None]:
    if not expected_columns or len(expected_columns) != len(set(expected_columns)):
        raise ValueError("The model must define a non-empty, unique feature list.")
    if customers.empty:
        raise ValueError("The customer CSV must contain at least one row.")
    if customers.columns.has_duplicates:
        raise ValueError("The customer CSV must not contain duplicate columns.")

    customer_ids = customers["customerID"].copy() if "customerID" in customers else None
    features = customers.drop(columns=["customerID"], errors="ignore")
    missing = sorted(set(expected_columns) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(expected_columns))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError(
            "Customer data does not match the model schema ("
            + "; ".join(details)
            + ")."
        )

    return features.loc[:, expected_columns], customer_ids
