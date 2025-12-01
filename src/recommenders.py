import pandas as pd

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

    # Filter by artist (multiple artists supported)
    def artist_matches(row_artists):
        return any(artist_name in a.strip() for a in row_artists.split(';'))

    song_idx = [i for i in song_idx if artist_matches(artist_lower.loc[i])]
    if len(song_idx) == 0:
        return None

    # Get first matched song
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

    # Exclude the input song
    cluster_songs = cluster_songs[cluster_songs.index != song.name]
    if cluster_songs.empty:
        return None

    recommendations = cluster_songs.sample(min(n, len(cluster_songs)))

    # Format artists for nicer display
    recommendations = recommendations.copy()
    recommendations['artists'] = recommendations['artists'].str.replace(';', ', ')

    return recommendations[['track_name', 'artists', 'album_name', 'popularity']]
