
# ============================================================
# CUSTOMER SEGMENTATION USING HIERARCHICAL CLUSTERING
# ============================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

df = pd.read_csv('customer_segmentation_dataset.csv')

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# STEP 3: DATA CLEANING
# ============================================================

# Preserve original customer segments for later comparison
original_segments = df['Segment'].copy()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# Remove identifier and existing segment labels
df = df.drop(columns=['CustomerID', 'Segment'])

# Handle missing values using column means
df['AnnualIncome_k'] = df['AnnualIncome_k'].fillna(
    df['AnnualIncome_k'].mean()
)

df['ReturnsRate'] = df['ReturnsRate'].fillna(
    df['ReturnsRate'].mean()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ============================================================
# STEP 4: FEATURE SELECTION
# ============================================================

X = df[
    [
        'Age',
        'AnnualIncome_k',
        'SpendingScore',
        'PurchasesPerYear',
        'AvgOrderValue',
        'AnnualRevenue',
        'DaysSinceLastPurchase',
        'LoyaltyYears',
        'ReturnsRate'
    ]
]

print("\nSelected Features:")
print(X.head())


# ============================================================
# STEP 5: FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Dataset Shape:")
print(X_scaled.shape)


# ============================================================
# STEP 6: CREATE HIERARCHICAL CLUSTERING MODEL
# ============================================================

hc_model = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'
)


# ============================================================
# STEP 7: TRAIN MODEL
# ============================================================

clusters = hc_model.fit_predict(X_scaled)

# Add cluster labels to dataset
df['Cluster'] = clusters

print("\nFirst 20 Cluster Assignments:")
print(clusters[:20])


# ============================================================
# STEP 8: CLUSTER ANALYSIS
# ============================================================

# Count customers in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()

print("\nCustomers in Each Cluster:")
print(cluster_counts)

# Calculate average feature values for each cluster
feature_cols = X.columns.tolist()

cluster_means = df.groupby('Cluster')[feature_cols].mean()

# Combine customer counts and feature averages
summary = cluster_means.copy()

summary.insert(0, 'Customer_Count', cluster_counts)

print("\nCluster Profile Summary:")
print(summary.round(2))


# ============================================================
# STEP 9: IMPROVED CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled[:, 1],       # Annual Income
    X_scaled[:, 2],       # Spending Score
    c=clusters,
    cmap='viridis',
    s=50,
    alpha=0.7
)

plt.title('Customer Segmentation Using Hierarchical Clustering')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.colorbar(label='Cluster')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 10: DENDROGRAM VISUALIZATION
# ============================================================

# Use a sample to make the dendrogram readable
sample_size = min(150, len(X_scaled))

np.random.seed(42)

sample_indices = np.random.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

# Generate hierarchical linkage matrix
linked = linkage(
    X_sample,
    method='ward',
    metric='euclidean'
)

plt.figure(figsize=(14, 7))

dendrogram(
    linked,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10
)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer Groups')
plt.ylabel('Euclidean Distance')
plt.tight_layout()
plt.show()


# ============================================================
# STEP 11: COMPARE DIFFERENT LINKAGE METHODS
# ============================================================

linkage_methods = ['ward', 'complete', 'average', 'single']

plt.figure(figsize=(14, 10))

for i, method in enumerate(linkage_methods, 1):

    model = AgglomerativeClustering(
        n_clusters=4,
        linkage=method
    )

    method_clusters = model.fit_predict(X_scaled)

    plt.subplot(2, 2, i)

    plt.scatter(
        X_scaled[:, 1],
        X_scaled[:, 2],
        c=method_clusters,
        cmap='viridis',
        s=35,
        alpha=0.7
    )

    plt.title(f'{method.capitalize()} Linkage')
    plt.xlabel('Annual Income (Scaled)')
    plt.ylabel('Spending Score (Scaled)')

plt.suptitle('Comparison of Hierarchical Linkage Methods', fontsize=15)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: COMPARE CLUSTERS WITH ORIGINAL SEGMENTS
# ============================================================

# Align original labels with cleaned dataset
original_segments_clean = original_segments.loc[df.index]

comparison = pd.crosstab(
    original_segments_clean,
    df['Cluster'],
    rownames=['Original Segment'],
    colnames=['Predicted Cluster']
)

print("\nOriginal Segments vs Predicted Clusters:")
print(comparison)


# ============================================================
# STEP 13: EVALUATE CLUSTER SIMILARITY
# ============================================================

# Convert original categories into numerical labels
original_labels = pd.factorize(original_segments_clean)[0]

# Adjusted Rand Index measures similarity between two partitions
ari = adjusted_rand_score(
    original_labels,
    clusters
)

print(f"\nAdjusted Rand Index: {ari:.4f}")


# ============================================================
# STEP 14: FINAL RESULTS
# ============================================================

print("\n========================================")
print("CUSTOMER SEGMENTATION COMPLETED")
print("========================================")

print(f"Total Customers: {len(df)}")
print(f"Total Clusters: {len(np.unique(clusters))}")

print("\nFinal Cluster Distribution:")
print(cluster_counts)

print("\nFinal Customer Profiles:")
print(summary.round(2))

# Save customer segmentation results
df.to_csv(
    'hierarchical_customer_segmentation_results.csv',
    index=False
)

print("\nResults saved successfully!")
