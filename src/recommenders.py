import pandas as pd
from sklearn.metrics import pairwise_distances

def recommend_by_song(song_name, df, artist_name, popularity=None, n=10):
    if not artist_name:
        raise ValueError("artist_name is required")

    # Standardize inputs
    song_name = song_name.lower().strip()
    artist_name = artist_name.lower().strip()

    # Lowercase for searching
    track_lower = df['track_name'].astype(str).str.lower().str.strip()
    artist_lower = df['artists'].astype(str).str.lower().str.strip()

    # Filter by song name
    song_idx = track_lower[track_lower.str.contains(song_name, na=False)].index
    if len(song_idx) == 0:
        return None

    # Filter by artist (supports multiple artists)
    def artist_matches(row_artists):
        return any(artist_name in a.strip() for a in row_artists.split(';'))

    song_idx = [i for i in song_idx if artist_matches(artist_lower.loc[i])]
    if len(song_idx) == 0:
        return None

    # Get first matched song
    song = df.loc[song_idx].iloc[0]
    cluster_id = song['cluster']

    # Select songs in the same cluster
    cluster_songs = df[df['cluster'] == cluster_id]

    # Optional popularity filter
    if popularity is not None:
        cluster_songs = cluster_songs[
            (cluster_songs['popularity'] >= popularity - 10) &
            (cluster_songs['popularity'] <= popularity + 10)
        ]
        if cluster_songs.empty:
            return None

    # Exclude input song
    cluster_songs = cluster_songs[cluster_songs.index != song.name]
    if cluster_songs.empty:
        return None

    # Remove duplicate songs
    cluster_songs = cluster_songs.drop_duplicates(subset=['track_name', 'artists'])

    # Compute distances to input song using audio features
    audio_features = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    song_features = song[audio_features].values.reshape(1, -1)
    cluster_features = cluster_songs[audio_features].values

    distances = pairwise_distances(song_features, cluster_features, metric='euclidean').flatten()

    # Select top n closest songs
    cluster_songs = cluster_songs.copy()
    cluster_songs['distance'] = distances
    recommendations = cluster_songs.nsmallest(n, 'distance')

    # Format artists nicely
    recommendations['artists'] = recommendations['artists'].str.replace(';', ', ')

    return recommendations[['track_name', 'artists', 'album_name', 'popularity']]