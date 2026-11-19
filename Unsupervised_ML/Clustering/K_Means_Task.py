import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

dataset =  load_iris(as_frame=True)
df = dataset.frame
X = dataset.data
print(df.head())
print(X.head())

# Step 3: Perform Exploratory Data Analysis (EDA)
print(df.head(5))
print(f'Shape of Dataset: {df.shape}')
print(f'Data Types of Columns: {df.dtypes}')
print(df.info())
print(f'Number of missing values: {df.isnull().sum()}')
print(f'Number of duplicated rows: {df.duplicated().sum()}')

df['species'] = dataset.target_names[dataset.target]

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, 
    x='petal length (cm)', 
    y='petal width (cm)', 
    hue='species',
    palette='viridis'
)
plt.title('Petal Length vs. Petal Width by Flower Species')
plt.show()

# Step 4: Select Features
X_matrix = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Step 5: Apply Feature Scaling
scalar = StandardScaler()
X_matrix_scaled = scalar.fit_transform(X_matrix)

# Step 6: Find the Optimal Number of Clusters Using Elbow Method.
# 1. Create an empty list to store inertia values
inertia_values = []
k_range = range(1, 11)

# 2. Use a loop to test K values from 1 to 10
for k in k_range:
    # 3. Initialize a new KMeans model for each K
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # 4. Fit the model on your scaled data
    kmeans.fit(X_matrix_scaled)
    
    # 5 & 6. Extract inertia and store it in the list
    inertia_values.append(kmeans.inertia_)

# 7. Plot K against inertia
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Step 8: Initialize and Train K-Means
K_M_Model = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)
K_M_Model.fit(X_matrix_scaled)

# Step 9: Extract Cluster Labels
labels = K_M_Model.labels_
print(f'Labels: {labels}')
df['Cluster'] = labels
print(df.head(5))
num_obs = df['Cluster'].value_counts().sort_index()
print(f'The Number of Observations in Each Cluster is: {num_obs}')

# Step 10: Extract Centroids
# 1. Extract centroid coordinates (in scaled feature space)
scaled_centroids = K_M_Model.cluster_centers_

# Put into a DataFrame for clear reading
scaled_centroids_df = pd.DataFrame(scaled_centroids, columns=dataset.feature_names)
print("Centroids in Scaled Feature Space:")
print(scaled_centroids_df)

# 3. Convert centroids back to the original feature scale (cm)
original_centroids = scalar.inverse_transform(scaled_centroids)

original_centroids_df = pd.DataFrame(original_centroids, columns=dataset.feature_names)
print("\nCentroids in Original Feature Scale (cm):")
print(original_centroids_df)

plt.figure(figsize=(9, 6))

# Step 11: Visualize Your Final Clusters
# 1. Plot the data points colored by their discovered cluster
sns.scatterplot(
    data=df,
    x='petal length (cm)',
    y='petal width (cm)',
    hue='Cluster',
    palette='viridis',
    s=70,
    alpha=0.8
)

# 2. Plot the cluster centroids on the same axes
sns.scatterplot(
    data=original_centroids_df,
    x='petal length (cm)',
    y='petal width (cm)',
    color='red',
    marker='X',
    s=250,
    edgecolor='black',
    linewidth=1.5,
    label='Centroids'
)

plt.title('K-Means Clustering of Iris Dataset (Petal Length vs. Petal Width)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Cluster / Center')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
