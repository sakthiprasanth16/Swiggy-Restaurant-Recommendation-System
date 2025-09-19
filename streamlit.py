import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Cache data loading
@st.cache_data
def load_data():
    for_encode_df = pd.read_csv("E:\\Sakthi\\prasanth\\projects\\swiggypro\\swiggy\\Scripts\\for_encode&cleaned.csv")
    encoded_df = pd.read_csv("E:\\Sakthi\\prasanth\\projects\\swiggypro\\swiggy\\Scripts\\encoded_data.csv")
    cleaned_data = pd.read_csv("E:\\Sakthi\\prasanth\\projects\\swiggypro\\swiggy\\Scripts\\cleaned_data.csv")
    return for_encode_df, encoded_df, cleaned_data

# Cache models
@st.cache_resource
def load_models():
    with open("E:\\Sakthi\\prasanth\\projects\\swiggypro\\swiggy\\Scripts\\city_encoder.pkl", "rb") as f:
        city_encoder = pickle.load(f)
    with open("E:\\Sakthi\\prasanth\\projects\\swiggypro\\swiggy\\Scripts\\cuisine_encoder.pkl", "rb") as f:
        cuisine_encoder = pickle.load(f)
    with open("E:\\Sakthi\\prasanth\\projects\\swiggypro\\swiggy\\Scripts\\scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return city_encoder, cuisine_encoder, scaler

# Cache KNN function
@st.cache_resource
def fit_knn(X):
    knn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(X)
    return knn

# Load all once
for_encode_df, encoded_df, cleaned_data = load_data()
city_encoder, cuisine_encoder, scaler = load_models()
knn = fit_knn(encoded_df)

# Streamlit UI
st.title("Swiggy Restaurant Recommendation System")

# City
city_options = for_encode_df['city'].dropna().unique().tolist()
city_options.insert(0, "None")
selected_city = st.selectbox("Select City", city_options)

# Cuisine
if selected_city != "None":
    cuisine_raw = for_encode_df[for_encode_df['city'] == selected_city]['cuisine'].dropna()
else:
    cuisine_raw = for_encode_df['cuisine'].dropna()

cuisine_full = cuisine_raw.unique()
cuisine_split = cuisine_raw.str.split(',').explode().str.strip().unique()
cuisine_options = np.unique(np.concatenate([cuisine_full, cuisine_split])).tolist()
cuisine_options.insert(0, "None")
selected_cuisine = st.selectbox("Select Cuisine", sorted(cuisine_options))

# Cost
if selected_cuisine != "None" and selected_city != "None":
    if "," in selected_cuisine:
        parts = [c.strip() for c in selected_cuisine.split(",")]
        cost_options = for_encode_df[
            (for_encode_df['city'] == selected_city) &
            (for_encode_df['cuisine'].str.contains("|".join(parts), case=False, na=False))
        ]['cost'].unique().tolist()
    else:
        cost_options = for_encode_df[
            (for_encode_df['city'] == selected_city) &
            (for_encode_df['cuisine'].str.contains(selected_cuisine, case=False, na=False))
        ]['cost'].unique().tolist()
else:
    cost_options = for_encode_df['cost'].dropna().unique().tolist()

cost_options.insert(0, "None")
selected_cost = st.selectbox(
    "Select Cost", 
    sorted(cost_options, key=lambda x: float(x) if x != "None" else -1)
)

# Rating
if selected_cuisine != "None" and selected_city != "None" and selected_cost != "None":
    if "," in selected_cuisine:
        parts = [c.strip() for c in selected_cuisine.split(",")]
        rating_options = for_encode_df[
            (for_encode_df['city'] == selected_city) &
            (for_encode_df['cost'] == selected_cost) &
            (for_encode_df['cuisine'].str.contains("|".join(parts), case=False, na=False))
        ]['rating'].unique().tolist()
    else:
        rating_options = for_encode_df[
            (for_encode_df['city'] == selected_city) &
            (for_encode_df['cost'] == selected_cost) &
            (for_encode_df['cuisine'].str.contains(selected_cuisine, case=False, na=False))
        ]['rating'].unique().tolist()
else:
    rating_options = for_encode_df['rating'].dropna().unique().tolist()

rating_options.insert(0, "None")
selected_rating = st.selectbox(
    "Select Rating",
    sorted(rating_options, key=lambda x: float(x) if x != "None" else -1)
)

# Recommendation Button
if st.button("Get Recommendations"):

    # Check if any input is None
    if selected_city == "None" or selected_cuisine == "None" or selected_cost == "None" or selected_rating == "None":
        st.warning("Please select City, Cuisine, Cost, and Rating before getting recommendations.")
    else:
        # Encode user input
        city_encoded = city_encoder.transform([[selected_city]])
        if "," in selected_cuisine:
            cuisine_encoded = cuisine_encoder.transform([selected_cuisine.split(',')])
        else:
            cuisine_encoded = cuisine_encoder.transform([[selected_cuisine]])

        rating_val = float(selected_rating)
        cost_val = float(selected_cost)
        rating_cost_scaled = scaler.transform(np.array([[rating_val, cost_val]]))

        # Combine features with using flatten(), it converts transformed 2D array to 1D array
        user_features = np.hstack([
            rating_cost_scaled.flatten(),
            city_encoded.flatten(),
            cuisine_encoded.flatten()
        ]).reshape(1, -1)

        # Find nearest neighbors
        distances, indices = knn.kneighbors(user_features)

        # Get recommendations
        recommendations = cleaned_data.iloc[indices[0]].copy()
        recommendations['Similarity'] = 1 - distances[0]

        # Filter the user input by city
        selected_city_lower = selected_city.strip().lower()
        recommendations['City'] = recommendations['City'].str.strip().str.lower()
        recommendations = recommendations[recommendations['City'] == selected_city_lower]

        # Display
        if not recommendations.empty:
            display_columns = ['Name', 'City', 'Cost', 'Cuisine', 'Rating', 'Rating Count', 'Address']
            recommendations_display = recommendations[display_columns].reset_index(drop=True)
            st.subheader("Recommended Restaurants")
            st.dataframe(recommendations_display, use_container_width=True)
        else:
            st.warning("No restaurants found for your selection in this city.")
