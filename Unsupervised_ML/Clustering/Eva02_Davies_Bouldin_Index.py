
# ============================================================
# PROJECT: CLUSTERING EVALUATION USING DAVIES-BOULDIN INDEX
# ============================================================


# STEP 1: IMPORT REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.mixture import GaussianMixture

from sklearn.metrics import davies_bouldin_score


# ============================================================
# STEP 2: GENERATE A SAMPLE DATASET
# ============================================================

# Generate a dataset containing 500 data points.
# We are creating 3 natural clusters.

X, y = make_blobs(
    n_samples=500,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

# Convert the dataset into a Pandas DataFrame.

df = pd.DataFrame(
    X,
    columns=['Feature_1', 'Feature_2']
)

print("First Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Visualize the original dataset before clustering.

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x='Feature_1',
    y='Feature_2'
)

plt.title("Original Dataset Before Clustering")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()


# ============================================================
# STEP 4: FEATURE SCALING
# ============================================================

# Distance-based clustering algorithms are sensitive
# to differences in feature scales.

# StandardScaler transforms the features so that they
# have approximately mean 0 and standard deviation 1.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Dataset:")
print(X_scaled[:5])


# ============================================================
# STEP 5: APPLY K-MEANS CLUSTERING
# ============================================================

# Initialize the K-Means model with 3 clusters.

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train the model and predict cluster labels.

labels = kmeans.fit_predict(X_scaled)

# Add the predicted cluster labels to the DataFrame.

df['Cluster'] = labels

print("\nCluster Assignments:")
print(df.head())

print("\nNumber of Data Points in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())


# ============================================================
# STEP 6: CALCULATE DAVIES-BOULDIN INDEX
# ============================================================

# davies_bouldin_score() calculates the Davies-Bouldin
# Index using the original scaled features and cluster labels.

dbi = davies_bouldin_score(
    X_scaled,
    labels
)

print("\n======================================")
print("      DAVIES-BOULDIN INDEX RESULT")
print("======================================")

print(f"Davies-Bouldin Index: {dbi:.4f}")


# ============================================================
# STEP 7: INTERPRET THE DAVIES-BOULDIN INDEX
# ============================================================

# Lower Davies-Bouldin Index indicates better clustering.
# The following thresholds are general guidelines only.

if dbi <= 0.5:

    print("Result: Very good clustering")

elif dbi <= 1.0:

    print("Result: Generally good clustering")

elif dbi <= 2.0:

    print("Result: Moderate clustering")

else:

    print("Result: Potentially poor clustering")


# ============================================================
# STEP 8: VISUALIZE THE CLUSTERS
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Feature_1',
    y='Feature_2',
    hue='Cluster',
    palette='viridis',
    s=60
)

# Transform the cluster centers back to the original scale.

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

# Display the cluster centers on the graph.

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=250,
    color='red',
    label='Centroids'
)

plt.title(
    f"K-Means Clustering (DBI: {dbi:.4f})"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()

plt.show()


# ============================================================
# STEP 9: FIND THE OPTIMAL NUMBER OF CLUSTERS
# ============================================================

# We will test different K values from 2 to 10.
# For each K value:
# 1. Apply K-Means.
# 2. Predict cluster labels.
# 3. Calculate DBI.
# 4. Store the result.

dbi_scores = []

k_values = range(2, 11)

for k in k_values:

    # Initialize K-Means with the current K value.

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Fit the model and predict cluster labels.

    cluster_labels = model.fit_predict(X_scaled)

    # Calculate Davies-Bouldin Index.

    current_dbi = davies_bouldin_score(
        X_scaled,
        cluster_labels
    )

    # Store the calculated DBI.

    dbi_scores.append(current_dbi)

    print(
        f"K = {k}, Davies-Bouldin Index = {current_dbi:.4f}"
    )


# ============================================================
# STEP 10: FIND THE BEST K VALUE
# ============================================================

# Unlike Silhouette Score, lower DBI is better.
# Therefore, we use np.argmin() to find the minimum score.

best_index = np.argmin(dbi_scores)

# Get the K value corresponding to the minimum DBI.

best_k = list(k_values)[best_index]

# Retrieve the lowest DBI score.

best_dbi = dbi_scores[best_index]

print("\n======================================")
print("         BEST CLUSTERING RESULT")
print("======================================")

print(f"Best Number of Clusters: {best_k}")

print(f"Lowest Davies-Bouldin Index: {best_dbi:.4f}")


# ============================================================
# STEP 11: VISUALIZE DBI FOR DIFFERENT K VALUES
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    dbi_scores,
    marker='o',
    linewidth=2
)

# Highlight the best K value.

plt.scatter(
    best_k,
    best_dbi,
    color='red',
    s=150,
    label=f'Best K = {best_k}'
)

plt.title("Davies-Bouldin Index vs Number of Clusters")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Davies-Bouldin Index")

plt.xticks(list(k_values))

plt.grid(True, alpha=0.3)

plt.legend()

plt.show()


# ============================================================
# STEP 12: COMPARE DIFFERENT CLUSTERING ALGORITHMS
# ============================================================

# We will compare four clustering algorithms:
#
# 1. K-Means
# 2. Hierarchical Clustering
# 3. DBSCAN
# 4. Gaussian Mixture Model
#
# Each algorithm will be evaluated using DBI.

results = []


# ------------------------------------------------------------
# A. K-MEANS CLUSTERING
# ------------------------------------------------------------

kmeans_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans_model.fit_predict(X_scaled)

kmeans_dbi = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

results.append({
    'Algorithm': 'K-Means',
    'DBI': kmeans_dbi
})


# ------------------------------------------------------------
# B. HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

hierarchical_model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

hierarchical_labels = hierarchical_model.fit_predict(
    X_scaled
)

hierarchical_dbi = davies_bouldin_score(
    X_scaled,
    hierarchical_labels
)

results.append({
    'Algorithm': 'Hierarchical',
    'DBI': hierarchical_dbi
})


# ------------------------------------------------------------
# C. DBSCAN CLUSTERING
# ------------------------------------------------------------

dbscan_model = DBSCAN(
    eps=0.5,
    min_samples=5
)

dbscan_labels = dbscan_model.fit_predict(X_scaled)

# DBSCAN assigns -1 to noise points.
# For this comparison, we exclude noise points from
# the DBI calculation.

non_noise_mask = dbscan_labels != -1

dbscan_valid_labels = dbscan_labels[non_noise_mask]

dbscan_valid_X = X_scaled[non_noise_mask]

# DBI requires at least 2 clusters and fewer clusters
# than the number of evaluated samples.

if (
    len(np.unique(dbscan_valid_labels)) >= 2
    and len(np.unique(dbscan_valid_labels)) < len(dbscan_valid_labels)
):

    dbscan_dbi = davies_bouldin_score(
        dbscan_valid_X,
        dbscan_valid_labels
    )

    results.append({
        'Algorithm': 'DBSCAN',
        'DBI': dbscan_dbi
    })

else:

    print("\nDBSCAN did not produce enough valid clusters.")
    print("DBI cannot be calculated for this configuration.")


# ------------------------------------------------------------
# D. GAUSSIAN MIXTURE MODEL
# ------------------------------------------------------------

gmm_model = GaussianMixture(
    n_components=3,
    random_state=42
)

gmm_labels = gmm_model.fit_predict(X_scaled)

gmm_dbi = davies_bouldin_score(
    X_scaled,
    gmm_labels
)

results.append({
    'Algorithm': 'Gaussian Mixture',
    'DBI': gmm_dbi
})


# ============================================================
# STEP 13: DISPLAY ALGORITHM COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Sort the algorithms by DBI in ascending order.
# The lowest DBI appears first.

results_df = results_df.sort_values(
    by='DBI',
    ascending=True
)

print("\n======================================")
print("       ALGORITHM COMPARISON")
print("======================================")

print(results_df.to_string(index=False))


# ============================================================
# STEP 14: VISUALIZE ALGORITHM COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x='Algorithm',
    y='DBI',
    hue='Algorithm',
    palette='viridis',
    legend=False
)

plt.title("Comparison of Clustering Algorithms Using DBI")

plt.xlabel("Clustering Algorithm")

plt.ylabel("Davies-Bouldin Index")

plt.xticks(rotation=20)

plt.grid(axis='y', alpha=0.3)

plt.show()


# ============================================================
# STEP 15: FINAL CONCLUSION
# ============================================================

print("\n======================================")
print("             FINAL CONCLUSION")
print("======================================")

print(f"Total Data Points: {len(X_scaled)}")

print(f"Best K Value: {best_k}")

print(f"Lowest DBI: {best_dbi:.4f}")

# Identify the algorithm with the lowest DBI.

best_algorithm = results_df.iloc[0]['Algorithm']

best_algorithm_score = results_df.iloc[0]['DBI']

print(f"Best Algorithm According to DBI: {best_algorithm}")

print(f"Its DBI Score: {best_algorithm_score:.4f}")

print("\nA lower Davies-Bouldin Index indicates")
print("more compact and better-separated clusters.")
print("Evaluation Technique Executed Successfully")
