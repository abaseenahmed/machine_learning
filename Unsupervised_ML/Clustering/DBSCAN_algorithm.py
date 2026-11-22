import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 1: Generate dataset
X, y = make_moons(
    n_samples=500,
    noise=0.08,
    random_state=42
)

# Step 2: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Initialize DBSCAN
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)

# Step 4: Fit the model
labels = dbscan.fit_predict(X_scaled)

# Step 5: Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis'
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 6: Count clusters and noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print("Number of Clusters:", n_clusters)
print("Number of Noise Points:", n_noise)