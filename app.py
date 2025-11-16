#  Import necessary libraries
import numpy as np # For numerical operations 
import pandas as pd # For data manipulation 
import matplotlib.pyplot as plt # 
import seaborn as sns # For advanced data visualization 
from sklearn.preprocessing import StandardScaler # For feature scaling 
from sklearn.cluster import KMeans # For K-Means clustering 
from sklearn.decomposition import PCA # For dimensionality reduction 
from sklearn.metrics import silhouette_score # For evaluating clustering performance 
import warnings # To suppress warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv('Mall_Customers.csv')

# print("Dataset Head:")
print(df.tail())
print(df.head())

df = df.dropna()
#  Select relevant features for segmentation
# We use 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
# Standardize the features (important for K-Means, as it is distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sns.pairplot(df[features])
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

wcss = []
cluster_range = range(1, 11)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
#  Plotting  the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster-Sum-of-Squares)')
plt.xticks(cluster_range)
plt.show()

optimal_clusters = 5  # Adjust this based on the Elbow Curve (e.g., where the "elbow" bends)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)   # Initialize K-Means with the optimal number of clusters
kmeans.fit(X_scaled)   # Fit the model to the scaled data
KMeans(n_clusters=5, random_state=42)

#  Add cluster labels to the original dataset
df['Cluster'] = kmeans.labels_
#  Step 6: Evaluate Clustering Performance
# Using the Silhouette Score to evaluate the quality of clustering
# The Silhouette Score ranges from -1 to 1, where higher values indicate better-defined clusters
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

pca = PCA(n_components=2)   # Initializes PCA to reduce features to 2 components
X_pca = pca.fit_transform(X_scaled)    # Transforms the scaled data into 2D

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
# Plotting clusters using a scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Segments (2D PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

cluster_summary = df.groupby('Cluster')[features].mean()
print("Cluster Summary:")
print(cluster_summary)
df.to_csv('customer_segmentation_results.csv', index=False)

print("Customer segmentation completed and results saved!")
print("This project was created by Nishant Dubey, CSE'26")