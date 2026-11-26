
# ============================================================
# PROJECT: EVALUATION OF CLUSTERING USING SILHOUETTE SCORE
# ============================================================


# STEP 1: IMPORT REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples
)

# ============================================================
# STEP 2: GENERATE A SAMPLE DATASET
# ============================================================

# Generate a dataset containing 500 data points.
# We are creating 3 natural clusters in this dataset.

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

print("First five rows of the dataset:")
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
# STEP 4: DATA SCALING
# ============================================================

# Scaling is important because distance-based clustering
# algorithms such as K-Means are affected by feature scales.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Dataset:")
print(X_scaled[:5])


# ============================================================
# STEP 5: APPLY K-MEANS CLUSTERING
# ============================================================

# We are selecting 3 clusters for our initial experiment.

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train the K-Means model and predict cluster labels.

labels = kmeans.fit_predict(X_scaled)

# Add the predicted cluster labels to the DataFrame.

df['Cluster'] = labels

print("\nCluster Assignments:")
print(df.head())

print("\nNumber of Data Points in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())


# ============================================================
# STEP 6: CALCULATE OVERALL SILHOUETTE SCORE
# ============================================================

# silhouette_score() calculates the average Silhouette
# Coefficient of all data points.

# It accepts:
# X_scaled = Scaled input features
# labels   = Cluster labels predicted by K-Means

score = silhouette_score(
    X_scaled,
    labels
)

print("\n======================================")
print("       SILHOUETTE SCORE RESULT")
print("======================================")

print(f"Silhouette Score: {score:.4f}")


# ============================================================
# STEP 7: INTERPRET THE SILHOUETTE SCORE
# ============================================================

# Interpret the score using general guidelines.

if score >= 0.70:

    print("Result: Excellent clustering")

elif score >= 0.50:

    print("Result: Good clustering")

elif score >= 0.25:

    print("Result: Weak or moderate clustering")

elif score >= 0:

    print("Result: Overlapping clusters")

else:

    print("Result: Potentially incorrect clustering")


# ============================================================
# STEP 8: CALCULATE INDIVIDUAL SILHOUETTE SCORES
# ============================================================

# silhouette_samples() calculates the Silhouette Score
# separately for every individual data point.

individual_scores = silhouette_samples(
    X_scaled,
    labels
)

# Add individual scores to the DataFrame.

df['Silhouette_Score'] = individual_scores

print("\nIndividual Silhouette Scores:")
print(df[['Cluster', 'Silhouette_Score']].head(10))


# Calculate the average Silhouette Score for each cluster.

cluster_scores = df.groupby('Cluster')[
    'Silhouette_Score'
].mean()

print("\nAverage Silhouette Score of Each Cluster:")
print(cluster_scores)


# ============================================================
# STEP 9: VISUALIZE THE CLUSTERS
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

# Display the cluster centers.

centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=250,
    color='red',
    label='Centroids'
)

plt.title(f"K-Means Clustering (Silhouette Score: {score:.4f})")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()
plt.show()


# ============================================================
# STEP 10: CREATE A SILHOUETTE PLOT
# ============================================================

# A Silhouette Plot displays the individual Silhouette
# Scores of all data points, grouped by their clusters.

fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10

n_clusters = len(np.unique(labels))

# Loop through every cluster.

for i in range(n_clusters):

    # Extract individual scores belonging to this cluster.

    cluster_silhouette_values = individual_scores[
        labels == i
    ]

    # Sort the scores for better visualization.

    cluster_silhouette_values.sort()

    # Calculate the number of data points in this cluster.

    cluster_size = cluster_silhouette_values.shape[0]

    # Calculate the upper boundary of the cluster section.

    y_upper = y_lower + cluster_size

    # Fill the silhouette area.

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        cluster_silhouette_values,
        alpha=0.7
    )

    # Add the cluster label.

    ax.text(
        -0.05,
        y_lower + cluster_size / 2,
        str(i)
    )

    # Prepare the starting position for the next cluster.

    y_lower = y_upper + 10


# Draw a vertical line showing the average Silhouette Score.

ax.axvline(
    x=score,
    color='red',
    linestyle='--',
    label=f'Average Score: {score:.4f}'
)

# Configure the plot.

ax.set_title("Silhouette Plot for K-Means Clustering")

ax.set_xlabel("Silhouette Coefficient Values")

ax.set_ylabel("Cluster Label")

ax.legend()

plt.show()


# ============================================================
# STEP 11: FIND THE OPTIMAL NUMBER OF CLUSTERS
# ============================================================

# We should not blindly assume that K=3 is the best choice.
# Therefore, we will test different values of K.

silhouette_scores = []

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

    # Calculate the Silhouette Score.

    current_score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    # Store the score.

    silhouette_scores.append(current_score)

    print(f"K = {k}, Silhouette Score = {current_score:.4f}")


# ============================================================
# STEP 12: FIND THE BEST K VALUE
# ============================================================

# Find the index of the highest Silhouette Score.

best_index = np.argmax(silhouette_scores)

# Retrieve the corresponding K value.

best_k = list(k_values)[best_index]

# Retrieve the highest Silhouette Score.

best_score = silhouette_scores[best_index]

print("\n======================================")
print("          BEST CLUSTERING")
print("======================================")

print(f"Best Number of Clusters: {best_k}")

print(f"Best Silhouette Score: {best_score:.4f}")


# ============================================================
# STEP 13: VISUALIZE SILHOUETTE SCORES FOR DIFFERENT K VALUES
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker='o',
    linewidth=2
)

# Highlight the best K value.

plt.scatter(
    best_k,
    best_score,
    color='red',
    s=150,
    label=f'Best K = {best_k}'
)

plt.title("Silhouette Score vs Number of Clusters")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Silhouette Score")

plt.xticks(list(k_values))

plt.grid(True, alpha=0.3)

plt.legend()

plt.show()


# ============================================================
# STEP 14: FINAL CONCLUSION
# ============================================================

print("\n========== FINAL CONCLUSION ==========")

print(f"Total Data Points: {len(X_scaled)}")

print(f"Best Number of Clusters: {best_k}")

print(f"Highest Silhouette Score: {best_score:.4f}")

print("The best K value among the tested values is:", best_k)
