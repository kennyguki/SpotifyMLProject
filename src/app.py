import streamlit as st
import pandas as pd
from joblib import load
from recommenders import recommend_by_song

# Load clustered dataset
df = pd.read_csv("data/clustered_dataset.csv")
scaler = load("models/scaler.pkl")
kmeans = load("models/kmeans.pkl")

st.title("Spotify Song Recommendation System")

# Song-based recommendation
st.header("Recommend by Song")
song_name = st.text_input("Song Name")
artist_name = st.text_input("Artist Name")

num_results = st.slider("Number of Recommendations", 1, 20, 10)

# Popularity slider (optional)
pop_min, pop_max = int(df['popularity'].min()), int(df['popularity'].max())
pop_value = st.slider("Optional Popularity", pop_min, pop_max, (pop_min + pop_max)//2)

# Checkbox to use popularity filter or not
use_pop = st.checkbox("Filter by popularity", value=False)

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
            st.table(recs)
    else:
        st.warning("Please enter both song name and artist.")