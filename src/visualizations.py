import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from joblib import load

# Load clustered dataset and models

df = pd.read_csv("data/clustered_dataset.csv")
scaler = load("models/scaler.pkl")
kmeans = load("models/kmeans.pkl")


audio_features = [
"danceability", "energy", "key", "loudness", "mode",
"speechiness", "acousticness", "instrumentalness",
"liveness", "valence", "tempo"
]

# -----------------------------

# Scale features for visualization

# -----------------------------

X_scaled = scaler.transform(df[audio_features])

# -----------------------------

# 1. PCA projection (2D) using scaled features

# -----------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='viridis', s=5)
plt.title("PCA Projection of Spotify Song Clusters (Scaled Features)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Cluster')
plt.show()

# -----------------------------

# 2. Scatter plot examples (optional: use scaled or original features)

# -----------------------------

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, audio_features.index('energy')],
X_scaled[:, audio_features.index('danceability')],
c=df['cluster'], cmap='viridis', s=5)
plt.xlabel("Energy (scaled)")
plt.ylabel("Danceability (scaled)")
plt.title("Energy vs Danceability by Cluster (Scaled)")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, audio_features.index('tempo')],
X_scaled[:, audio_features.index('valence')],
c=df['cluster'], cmap='viridis', s=5)
plt.xlabel("Tempo (scaled)")
plt.ylabel("Valence (scaled)")
plt.title("Tempo vs Valence by Cluster (Scaled)")
plt.show()

# -----------------------------

# 3. Number of songs per cluster

# -----------------------------

df['cluster'].value_counts().sort_index().plot(kind='bar', figsize=(10,4))
plt.title("Number of Songs per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# -----------------------------

# 4. Cluster centroids heatmap (scaled features)

# -----------------------------

cluster_centers_scaled = np.zeros((df['cluster'].nunique(), X_scaled.shape[1]))
for i in range(df['cluster'].nunique()):
    cluster_centers_scaled[i] = X_scaled[df['cluster'] == i].mean(axis=0)

plt.figure(figsize=(12,6))
sns.heatmap(cluster_centers_scaled, cmap='coolwarm', annot=True, xticklabels=audio_features)
plt.title("Cluster Centroids (Scaled Features)")
plt.xlabel("Audio Feature")
plt.ylabel("Cluster")
plt.show()