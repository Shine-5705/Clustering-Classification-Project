import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = pd.read_csv("DATASETS/train - train.csv")
X = data.drop("target", axis=1)
y = data["target"]  # Assuming target is for evaluation

# Reduce dimensionality with PCA (7 components)
pca = PCA(n_components=7)
pca_data = pca.fit_transform(X)

# Determine optimal number of clusters (wider range for potentially more clusters)
silhouette_scores = []
for k in range(2, 21):  # Explore up to 20 clusters
   kmeans = KMeans(n_clusters=k)
   kmeans.fit(pca_data)
   silhouette_scores.append(silhouette_score(pca_data, kmeans.labels_))
optimal_k = np.argmax(silhouette_scores) + 2  # Adjust for starting at 2

# Perform clustering with the potentially higher number of clusters
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(pca_data)

# Predict clusters for new data point (using PCA for consistency)
new_data_1 = [-68,-61,-64,-59,-58,-70,-67,-76,-73,-69,-72,-74,-54,-68,-69,-75,-76,-95]
new_point = np.array([new_data_1])
predicted_cluster = kmeans.predict(pca.transform(new_point))[0]

# Print or store the predicted cluster
print(f"Predicted cluster for new data point: {predicted_cluster}")

# Visualize clusters (using first two PCA components for illustration)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clusters with PCA (7 Components)")
plt.show()
