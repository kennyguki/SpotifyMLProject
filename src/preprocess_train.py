import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Select audio features for clustering
audio_features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]

X = df[audio_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered dataset
df.to_csv("data/clustered_dataset.csv", index=False)

print("Clustering complete. Dataset saved with cluster labels.")

# Example plots to view clusters and how features relate to each other

plt.scatter(df['energy'], df['danceability'], c=df['cluster'], cmap='viridis')
plt.xlabel("Energy")
plt.ylabel("Danceability")
plt.title("K-Means Clusters of Spotify Songs")
plt.show()