import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

@st.cache
def load_data():
    data = pd.read_csv("/home/shinegupta/Documents/deployement/train - train.csv")
    if data.columns is None:
        data.columns = list(range(data.shape[1]))  
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data...done!')

X = data.drop("target", axis=1)
y = data["target"]

# Reduce dimensionality with PCA (2 components)
pca = PCA(n_components=2)  # Reduced to 2 components for visualization
pca_data = pca.fit_transform(X)

# Determine optimal number of clusters
optimal_k = 5  # Assume optimal number of clusters is known

# Perform clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(pca_data)

# User input section
st.header("Predict Cluster for New Data Point")
new_data_features = []
for i in range(1, X.shape[1] + 1):
    feature = st.number_input(f"Enter feature {i}:", value=0.0)
    new_data_features.append(feature)

if st.button('Submit'):
    new_point = np.array([new_data_features])
    predicted_cluster = kmeans.predict(pca.transform(new_point))[0]

    # Print predicted cluster (Highlighted Output)
    st.markdown(f"<p style='font-size:20px; font-weight:bold;'>Predicted cluster for new data point: {predicted_cluster}</p>", unsafe_allow_html=True)

    # Visualize clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_, cmap='viridis')

    # Highlight new data point
    ax.scatter(pca.transform(new_point)[:, 0], pca.transform(new_point)[:, 1], color='red', label='New Point', s=100, marker='x')

    # Highlight cluster of the new data point
    cluster_center = kmeans.cluster_centers_[predicted_cluster]
    ax.scatter(cluster_center[0], cluster_center[1], color='red', marker='o', s=200, label='Predicted Cluster Center')
    ax.annotate(f'Predicted Cluster: {predicted_cluster}', xy=(cluster_center[0], cluster_center[1]), xytext=(cluster_center[0] + 1, cluster_center[1] + 1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-Means Clusters with PCA (2 Components)")
    ax.legend()
    st.pyplot(fig)
