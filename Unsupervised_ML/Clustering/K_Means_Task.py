"""Explore K-Means clustering on the Iris dataset.

The script reports a compact data overview, compares candidate cluster counts,
and saves polished visualizations in the adjacent ``reports`` directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
# Keep three clusters as a familiar Iris demonstration baseline.
SELECTED_CLUSTERS = 3
MAX_CLUSTERS = 10
REPORTS_DIR = Path(__file__).parent / "reports"


def configure_plot_style() -> None:
    """Apply consistent styling and export quality to every chart."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelweight": "medium",
            "grid.alpha": 0.25,
        }
    )


def load_iris_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the Iris feature table and a labeled copy for visualization."""
    iris = load_iris(as_frame=True)
    features = iris.data.copy()
    data = iris.frame.copy()
    data["species"] = iris.target.map(dict(enumerate(iris.target_names)))
    data = data.drop(columns="target")
    return features, data


def print_data_overview(features: pd.DataFrame, data: pd.DataFrame) -> None:
    """Print a concise overview of the dataset and its feature quality."""
    print("\n========== IRIS DATASET OVERVIEW ==========")
    print(f"Observations: {len(features)}")
    print(f"Features: {', '.join(features.columns)}")
    print("\nFirst five observations:")
    print(data.head().to_string(index=False))
    print("\nFeature data types:")
    print(features.dtypes.to_string())
    print(f"\nMissing feature values: {features.isna().sum().sum()}")
    print(f"Duplicate observations: {features.duplicated().sum()}")
    print("\nFeature summary:")
    print(features.describe().round(2).to_string())


def save_and_show(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the reports directory, display it, and release it."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.show()
    plt.close(fig)


def plot_species_distribution(data: pd.DataFrame) -> None:
    """Visualize petal measurements by the known Iris species."""
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    sns.scatterplot(
        data=data,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        palette="colorblind",
        s=75,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set(
        title="Iris Species: Petal Measurement Distribution",
        xlabel="Petal length (cm)",
        ylabel="Petal width (cm)",
    )
    ax.legend(title="Species", frameon=True)
    save_and_show(fig, "iris_species_distribution.png")


def compare_cluster_counts(
    scaled_features: pd.DataFrame,
) -> tuple[list[int], list[float], list[float]]:
    """Calculate inertia and silhouette scores for candidate cluster counts."""
    max_clusters = min(MAX_CLUSTERS, len(scaled_features) - 1)
    elbow_candidates = list(range(1, max_clusters + 1))
    silhouette_candidates = list(range(2, max_clusters + 1))
    inertia_values: list[float] = []
    silhouette_values: list[float] = []

    for cluster_count in elbow_candidates:
        model = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(scaled_features)
        inertia_values.append(model.inertia_)

        if cluster_count >= 2:
            silhouette_values.append(
                silhouette_score(scaled_features, labels)
            )

    return elbow_candidates, inertia_values, silhouette_values


def plot_cluster_comparison(
    cluster_counts: list[int],
    inertia_values: list[float],
    silhouette_values: list[float],
) -> None:
    """Plot inertia and silhouette scores side by side for model selection."""
    silhouette_counts = cluster_counts[1:]
    best_count = silhouette_counts[silhouette_values.index(max(silhouette_values))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

    axes[0].plot(
        cluster_counts,
        inertia_values,
        marker="o",
        linewidth=2,
        color=sns.color_palette("colorblind")[0],
    )
    axes[0].axvline(
        SELECTED_CLUSTERS,
        linestyle="--",
        color="#555555",
        label=f"Demonstration K = {SELECTED_CLUSTERS}",
    )
    axes[0].set(
        title="Elbow Method",
        xlabel="Number of clusters (K)",
        ylabel="Inertia (within-cluster sum of squares)",
        xticks=cluster_counts,
    )
    axes[0].legend(frameon=True)

    bar_colors = [
        sns.color_palette("colorblind")[2] if count == best_count
        else sns.color_palette("colorblind")[1]
        for count in silhouette_counts
    ]
    axes[1].bar(
        silhouette_counts,
        silhouette_values,
        color=bar_colors,
        edgecolor="white",
        label=f"Highest score (K = {best_count})",
    )
    axes[1].axvline(
        SELECTED_CLUSTERS,
        linestyle="--",
        color="#555555",
        label=f"Demonstration K = {SELECTED_CLUSTERS}",
    )
    axes[1].set(
        title="Silhouette Score",
        xlabel="Number of clusters (K)",
        ylabel="Mean silhouette score",
        xticks=silhouette_counts,
        ylim=(0, 1),
    )
    axes[1].legend(frameon=True)

    fig.suptitle("Comparing Candidate K-Means Models", fontsize=15, weight="bold")
    save_and_show(fig, "kmeans_cluster_comparison.png")


def fit_final_model(
    features: pd.DataFrame,
    scaled_features: pd.DataFrame,
    scaler: StandardScaler,
) -> tuple[KMeans, pd.DataFrame, pd.DataFrame]:
    """Fit the selected model and return labeled data and original-scale centers."""
    model = KMeans(
        n_clusters=SELECTED_CLUSTERS,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(scaled_features)
    clustered_data = features.copy()
    clustered_data["Cluster"] = labels

    scaled_centers = pd.DataFrame(
        model.cluster_centers_,
        columns=features.columns,
    )
    original_centers = pd.DataFrame(
        scaler.inverse_transform(scaled_centers),
        columns=features.columns,
    )
    return model, clustered_data, original_centers


def print_model_summary(
    model: KMeans,
    clustered_data: pd.DataFrame,
    original_centers: pd.DataFrame,
    scaled_features: pd.DataFrame,
) -> None:
    """Display cluster sizes, centroids, and a model-quality summary."""
    print("\n========== CLUSTERING RESULTS ==========")
    print(
        "The demonstration keeps K = 3; use the comparison chart "
        "to evaluate alternatives."
    )
    print("Cluster sizes:")
    print(clustered_data["Cluster"].value_counts().sort_index().to_string())
    print("\nCentroids in original feature units:")
    print(original_centers.round(2).to_string(index=False))
    print(f"\nInertia: {model.inertia_:.3f}")
    print(f"Silhouette score: {silhouette_score(scaled_features, model.labels_):.3f}")
    print(f"Iterations to converge: {model.n_iter_}")


def plot_final_clusters(
    clustered_data: pd.DataFrame,
    original_centers: pd.DataFrame,
) -> None:
    """Show the final two-feature cluster view and projected centroids."""
    x_feature = "petal length (cm)"
    y_feature = "petal width (cm)"
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    sns.scatterplot(
        data=clustered_data,
        x=x_feature,
        y=y_feature,
        hue="Cluster",
        hue_order=sorted(clustered_data["Cluster"].unique()),
        palette="colorblind",
        s=75,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.scatter(
        original_centers[x_feature],
        original_centers[y_feature],
        marker="X",
        s=250,
        color="#222222",
        edgecolor="white",
        linewidth=1.2,
        label="Centroids",
        zorder=3,
    )
    ax.set(
        title="K-Means Clusters on Iris Petal Measurements",
        xlabel="Petal length (cm)",
        ylabel="Petal width (cm)",
    )
    ax.legend(title="Cluster", frameon=True)
    save_and_show(fig, "kmeans_final_clusters.png")


def main() -> None:
    """Run the Iris clustering analysis from data inspection to visualization."""
    configure_plot_style()
    features, data = load_iris_data()
    print_data_overview(features, data)
    plot_species_distribution(data)

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    cluster_counts, inertia_values, silhouette_values = compare_cluster_counts(
        scaled_features
    )
    plot_cluster_comparison(
        cluster_counts,
        inertia_values,
        silhouette_values,
    )

    model, clustered_data, original_centers = fit_final_model(
        features,
        scaled_features,
        scaler,
    )
    print_model_summary(
        model,
        clustered_data,
        original_centers,
        scaled_features,
    )
    plot_final_clusters(clustered_data, original_centers)


if __name__ == "__main__":
    main()
