
# ============================================================
# PROJECT: CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ALGORITHM: K-MEANS (UNSUPERVISED LEARNING)
# ============================================================

# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# STEP 2: CREATE DATASET

data = {
    "Annual_Income": [
        15, 18, 20, 22, 25,
        50, 55, 58, 60, 65,
        85, 90, 95, 100, 105
    ],

    "Spending_Score": [
        20, 25, 22, 28, 30,
        50, 55, 52, 58, 60,
        80, 85, 82, 90, 95
    ]
}

df = pd.DataFrame(data)

print("\n========== ORIGINAL DATASET ==========")
print(df)


# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE VALUES ==========")
print(df.duplicated().sum())


# STEP 4: VISUALIZE ORIGINAL DATA

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    s=100
)

plt.title("Original Customer Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# STEP 5: SELECT FEATURES

X = df[["Annual_Income", "Spending_Score"]]

print("\n========== SELECTED FEATURES ==========")
print(X.head())


# STEP 6: FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== SCALED DATA ==========")
print(X_scaled[:5])


# STEP 7: FIND OPTIMAL K USING ELBOW METHOD

inertia_values = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)


# VISUALIZE ELBOW METHOD

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# STEP 8: EVALUATE DIFFERENT K VALUES USING SILHOUETTE SCORE

print("\n========== SILHOUETTE SCORES ==========")

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print(f"K = {k}, Silhouette Score = {score:.4f}")


# STEP 9: TRAIN FINAL K-MEANS MODEL

# Based on the natural three groups in this demonstration dataset.
# In real projects, evaluate K using the previous two steps.

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(X_scaled)


# STEP 10: GET CLUSTER LABELS

labels = kmeans.labels_

df["Cluster"] = labels

print("\n========== CLUSTERED DATASET ==========")
print(df)


# STEP 11: GET CENTROIDS

centroids = kmeans.cluster_centers_

print("\n========== CLUSTER CENTROIDS (SCALED) ==========")
print(centroids)

# Convert centroids back to original feature scales

original_centroids = scaler.inverse_transform(centroids)

centroid_df = pd.DataFrame(
    original_centroids,
    columns=["Annual_Income", "Spending_Score"]
)

print("\n========== CENTROIDS (ORIGINAL SCALE) ==========")
print(centroid_df)


# STEP 12: CLUSTER SIZE

print("\n========== NUMBER OF CUSTOMERS IN EACH CLUSTER ==========")

print(df["Cluster"].value_counts().sort_index())


# STEP 13: VISUALIZE FINAL CLUSTERS

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="Cluster",
    palette="viridis",
    s=120
)

# Plot cluster centroids in original feature space

plt.scatter(
    original_centroids[:, 0],
    original_centroids[:, 1],
    marker="X",
    s=300,
    color="red",
    label="Centroids"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)
plt.show()


# STEP 14: FINAL MODEL EVALUATION

print("\n========== MODEL EVALUATION ==========")

print("Inertia:", kmeans.inertia_)

print("Number of Iterations:", kmeans.n_iter_)

print(
    "Silhouette Score:",
    silhouette_score(X_scaled, kmeans.labels_)
)


# STEP 15: PREDICT CLUSTER FOR A NEW CUSTOMER

# New customer:
# Annual Income = 60
# Spending Score = 55

new_customer = pd.DataFrame(
    [[60, 55]],
    columns=["Annual_Income", "Spending_Score"]
)

# Apply the SAME scaler used during training

new_customer_scaled = scaler.transform(new_customer)

# Predict the customer's cluster

prediction = kmeans.predict(new_customer_scaled)

print("\n========== NEW CUSTOMER PREDICTION ==========")

print("Customer Information:")
print(new_customer)

print("Assigned Cluster:", prediction[0])

print("\n========== PROJECT COMPLETED ==========")
