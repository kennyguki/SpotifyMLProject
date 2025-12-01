import streamlit as st
import pandas as pd
from joblib import load
from recommenders import recommend_by_song
import os


# Load clustered dataset and models
df = pd.read_csv("data/clustered_dataset.csv")

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")  # go up one level

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
kmeans_path = os.path.join(MODELS_DIR, "kmeans.pkl")

scaler = load(scaler_path)
kmeans = load(kmeans_path)

st.set_page_config(page_title="Music Recommendation System", layout="wide")

# Sidebar
st.sidebar.image("data/spotify-logo-spotify-symbol-3.png", width='stretch')  # Updated parameter

st.sidebar.markdown("## 🎧 About this App")
st.sidebar.info(
    """
    Enter a **song name** and its **artist** to get similar song recommendations 
    based on Spotify audio features. 🎶  

    Optionally, filter by popularity to refine your results.
    
    Note: Last dataset update was in **2024**, new releases (and some older) may not be included.
    """
)

st.sidebar.markdown("---")  # Separator line

st.sidebar.markdown(
    "💡 **Tip:** Try entering your favorite song and artist for a curated playlist!"
)

st.sidebar.markdown("---") # Separator line

st.sidebar.markdown("## 🔗 Useful Links")
st.sidebar.markdown(
    """
    - [Spotify](https://spotify.com)
    - [GitHub Repo](https://github.com/kennyguki/SpotifyMLProject)
    - [Dataset](https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset)  
    - [LinkedIn](https://www.linkedin.com/in/kennethgutierrezking/)  
    """
)

st.title("🎵 Music Recommendation System")

# Input section in columns
col1, col2 = st.columns(2)
with col1:
    song_name = st.text_input("Song Name")
with col2:
    artist_name = st.text_input("Artist Name")

# Number of results
num_results = st.slider("Number of Recommendations", 1, 20, 10)

# Popularity filter
use_pop = st.checkbox("🎯 Filter by popularity (Recommended)")
if use_pop:
    st.markdown("🎵 Adjust popularity range by ±10 from your selected song")
    pop_min, pop_max = int(df['popularity'].min()), int(df['popularity'].max())
    pop_value = st.slider("Popularity", pop_min, pop_max, (pop_min + pop_max)//2)

# Recommendation button
if st.button("Get Recommendations"):
    if song_name.strip() and artist_name.strip():
        recs = recommend_by_song(
            song_name,
            df,
            artist_name,
            popularity=pop_value if use_pop else None,
            n=num_results
        )
        if recs is None or recs.empty:
            st.warning("Song not found or no recommendations in that cluster.")
        else:
            # Format table for display
            recs_display = recs.copy()
            
            # Rename columns for cleaner display
            recs_display = recs_display.rename(columns={
                'track_name': 'Song',
                'artists': 'Artist(s)',
                'album_name': 'Album',
                'popularity': 'Popularity'
            })
            
            # Capitalize text for consistency
            recs_display['Song'] = recs_display['Song'].str.title()
            recs_display['Artist(s)'] = recs_display['Artist(s)'].str.title()
            recs_display['Album'] = recs_display['Album'].str.title()
            
            # Add popularity emojis
            recs_display['Popularity'] = recs_display['Popularity'].apply(
                lambda x: f"{x} {'🔥' if x >= 80 else '⭐' if x >= 70 else ''}"
            )
            
            st.success("Recommended Songs 🎶")
            st.dataframe(recs_display.reset_index(drop=True), width='stretch')
    else:
        st.warning("Please enter both song name and artist.")