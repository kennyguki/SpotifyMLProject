import pandas as pd

def recommend_by_song(song_name, df, artist_name, popularity=None, n=10):
    if not artist_name:
        raise ValueError("artist_name is required")

    # Standardize inputs
    song_name = song_name.lower().strip()
    artist_name = artist_name.lower().strip()

    # Use temporary lowercased Series for searching — don't mutate df
    track_lower = df['track_name'].astype(str).str.lower().str.strip()
    artist_lower = df['artists'].astype(str).str.lower().str.strip()

    # Filter by song name
    song_idx = track_lower[track_lower.str.contains(song_name, na=False)].index
    if song_idx.empty:
        return None

    # Filter by artist
    song_idx = song_idx[artist_lower.loc[song_idx].str.contains(artist_name, na=False)]
    if len(song_idx) == 0:
        return None

    # Get first matched row (keep original casing)
    song = df.loc[song_idx].iloc[0]
    cluster_id = song['cluster']
    cluster_songs = df[df['cluster'] == cluster_id]

    # Optional popularity filter
    if popularity is not None:
        cluster_songs = cluster_songs[
            (cluster_songs['popularity'] >= popularity - 10) &
            (cluster_songs['popularity'] <= popularity + 10)
        ]
        if cluster_songs.empty:
            return None

    # Sample recommendations (returns original-case strings)
    recommendations = cluster_songs.sample(min(n, len(cluster_songs)))
    return recommendations[['track_name', 'artists', 'album_name', 'popularity']]
