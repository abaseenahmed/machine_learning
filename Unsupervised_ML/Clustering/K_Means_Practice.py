# Step 01: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 02: Create or Load a Dataset
data = {
    'Annual_Income': [
        15, 18, 20, 22, 25,
        50, 55, 58, 60, 65,
        85, 90, 95, 100, 105
    ],

    'Spending_Score': [
        20, 25, 22, 28, 30,
        50, 55, 52, 58, 60,
        80, 85, 82, 90, 95
    ]
}

df = pd.DataFrame(data)

print(df.head())

# Step 03: Perform Exploratory Data Analysis (EDA)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x='Annual_Income',
    y='Spending_Score'
)

plt.title('Customer Distribution')
plt.show()

# Step 04: Select the Features
# X contains our independent features.
# This is one of the main differences between supervised and unsupervised learning.

X = df[['Annual_Income', 'Spending_Score']]

# Step 05: Feature Scaling (Very Important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 06: Apply the K - Means Clustering algorithm
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(X_scaled)

# Step 07: Get the cluster labels 
labels = kmeans.labels_
print(labels)
df['Cluster'] = labels
print(df.head())

# Step 08: Visualize the Final Clusters
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Annual_Income',
    y='Spending_Score',
    hue='Cluster',
    palette='viridis',
    s=100
)

plt.title('Customer Segmentation Using K-Means')

plt.show()

# Step 09: Predict New data items.

centers = kmeans.cluster_centers_
print(centers)
print(kmeans.inertia_)

new_customer = [[60, 55]]
new_customer_scaled = scaler.transform(new_customer)
prediction = kmeans.predict(new_customer_scaled)
print(prediction)

