
# Step 1: Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Step 2: Generate sample dataset

X, y = make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=1.5,
    random_state=42
)

# Step 3: Scale the data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Initialize GMM model

gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42
)

# Step 5: Train the model

gmm.fit(X_scaled)

# Step 6: Predict cluster labels

labels = gmm.predict(X_scaled)

# Step 7: Calculate cluster probabilities

probabilities = gmm.predict_proba(X_scaled)

# Step 8: Add results to DataFrame

df = pd.DataFrame(X_scaled, columns=['Feature_1', 'Feature_2'])

df['Cluster'] = labels

print(df.head())

# Step 9: Display probabilities

print("\nCluster Probabilities:")
print(probabilities[:5])

# Step 10: Visualize clusters

plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis',
    s=40
)

plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
