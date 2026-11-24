"""Apply DBSCAN to this directory's realistic, synthetic customer dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).with_name("customer_segmentation_dataset.csv")
FEATURE_COLUMNS = ["AnnualIncome_k", "SpendingScore", "DaysSinceLastPurchase"]
EPS = 0.6
MIN_SAMPLES = 10


def load_customer_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and check that all clustering features are present."""
    data = pd.read_csv(data_path)
    missing_columns = set(FEATURE_COLUMNS).difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required feature columns: {missing}")
    return data


def main() -> None:
    data = load_customer_data()

    # Keep identifiers and the existing Segment label out of unsupervised training.
    features = data.loc[:, FEATURE_COLUMNS]
    imputed_features = SimpleImputer(strategy="median").fit_transform(features)
    scaled_features = StandardScaler().fit_transform(imputed_features)

    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    cluster_labels = model.fit_predict(scaled_features)
    data["Cluster"] = cluster_labels

    cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = int(np.count_nonzero(cluster_labels == -1))

    print(f"Dataset: {DATA_PATH.name}")
    print(f"Customers: {len(data)}")
    print(f"Features used: {', '.join(FEATURE_COLUMNS)}")
    print(f"DBSCAN parameters: eps={EPS}, min_samples={MIN_SAMPLES}")
    print(f"Clusters found: {cluster_count}")
    print(f"Noise points: {noise_count}")
    print("\nCustomer count by cluster (-1 represents noise):")
    print(data["Cluster"].value_counts().sort_index().to_string())

    clustered_customers = data.loc[data["Cluster"] != -1]
    if not clustered_customers.empty:
        cluster_profile = clustered_customers.groupby("Cluster")[FEATURE_COLUMNS].mean()
        cluster_profile.insert(
            0,
            "CustomerCount",
            clustered_customers.groupby("Cluster").size(),
        )
        print("\nAverage feature values for each cluster:")
        print(cluster_profile.round(2).to_string())
    else:
        print("\nDBSCAN did not form any clusters; all customers were labeled as noise.")

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        imputed_features[:, 0],
        imputed_features[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=35,
        alpha=0.75,
    )
    plt.title("DBSCAN Customer Clusters")
    plt.xlabel("Annual Income (thousands)")
    plt.ylabel("Spending Score")
    plt.colorbar(scatter, label="Cluster label (-1 = noise)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
