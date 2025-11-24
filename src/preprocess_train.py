import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Select audio features for clustering
audio_features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

X = df[audio_features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means
kmeans = KMeans(n_clusters=50, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered dataset
df.to_csv("data/clustered_dataset.csv", index=False)

# Create the models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")

print("Clustering complete. Dataset saved with cluster labels and model saved.")