# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# 2. Generate sample dataset
X, y = make_blobs(
    n_samples=200,
    centers=3,
    random_state=42
)

# 3. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Create the model
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

# 5. Train and predict clusters
clusters = model.fit_predict(X_scaled)

# 6. Visualize results
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters,
    cmap='viridis'
)

plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()