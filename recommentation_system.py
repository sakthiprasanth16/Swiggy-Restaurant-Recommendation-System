import streamlit as st
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Swiggy Restaurant Recommendation", layout="wide")
st.title("Swiggy Restaurant Recommendation")

BASE_PATH = Path(r"E:\Sakthi\prasanth\projects\swiggypro\swiggy\Scripts")
CLEANED_CSV = BASE_PATH / "cleaned_data.csv"
ENCODED_CSV = BASE_PATH / "encode_data.csv"
SCALER_PATH = BASE_PATH / "scaler.pkl"
OHE_PATH = BASE_PATH / "city_encoder.pkl"
MLB_PATH = BASE_PATH / "cuisine_encoder.pkl"

def load_pickle_path(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def read_csv_path(path: Path):
    return pd.read_csv(path)

def extract_city_parts(user_city_str: str):
    # expects "city,main_city" or "city", returns (city, main_city)
    if not isinstance(user_city_str, str) or user_city_str.strip() == "":
        return ("", "0")
    s = user_city_str.strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        return (parts[0], "0")
    else:
        return (parts[0], parts[-1])

def parse_raw_cuisine_strings(raw_strings):
    """
    Accepts a list of raw cuisine strings (each possibly comma-separated)
    and returns a deduped ordered list of cuisine tokens (lower-cased, stripped).
    """
    toks = []
    seen = set()
    for s in raw_strings:
        if not isinstance(s, str) or not s.strip():
            continue
        for part in s.split(","):
            token = part.strip().lower()
            if token and token not in seen:
                seen.add(token)
                toks.append(token)
    return toks

# Check files exist
missing = [p for p in [CLEANED_CSV, ENCODED_CSV, SCALER_PATH, OHE_PATH, MLB_PATH] if not p.exists()]
if missing:
    st.error("Missing required files in BASE_PATH. Check the following are present:")
    st.write([str(p) for p in missing])
    st.stop()

# Load data
try:
    df_clean = read_csv_path(CLEANED_CSV)
except Exception as e:
    st.error(f"Failed to read cleaned_data.csv: {e}")
    st.stop()

try:
    df_encoded = read_csv_path(ENCODED_CSV)
except Exception as e:
    st.error(f"Failed to read encode_data.csv: {e}")
    st.stop()

try:
    scaler = load_pickle_path(SCALER_PATH)
    ohe = load_pickle_path(OHE_PATH)
    mlb = load_pickle_path(MLB_PATH)
except Exception as e:
    st.error(f"Failed to load one or more pickle artifacts: {e}")
    st.stop()

# Option lists from cleaned_data
# City examples: build strings like "city,MainCity" when available, else "city"
city_area_main_options = []
if {"city", "main_city"}.issubset(df_clean.columns):
    pairs = df_clean[["city", "main_city"]].fillna("").astype(str)
    for _, r in pairs.iterrows():
        area = r["city"].strip()
        main = r["main_city"].strip()
        if area and main and main != "0":
            city_area_main_options.append(f"{area},{main}")
        elif area:
            city_area_main_options.append(area)
    city_area_main_options = sorted(set(city_area_main_options))
else:
    if "city" in df_clean.columns:
        city_area_main_options = sorted(df_clean["city"].dropna().astype(str).unique().tolist())

# Cuisine options
cuisine_raw_options = []
if "cuisine" in df_clean.columns:
    cuisine_raw_options = sorted(df_clean["cuisine"].dropna().astype(str).unique().tolist())

# Ratings options
rating_options = []
if "rating" in df_clean.columns:
    rating_series = pd.to_numeric(df_clean["rating"], errors="coerce").dropna()
    rating_options = sorted(list(rating_series.unique()))

cost_options = []
if "cost" in df_clean.columns:
    cost_series = pd.to_numeric(df_clean["cost"], errors="coerce").dropna()
    cost_options = sorted(list(cost_series.unique()))    

# Prepare numeric matrix for encoded data
try:
    X = df_encoded.values.astype(np.float32)  # encoded features matrix
except Exception as e:
    st.error(f"encode_data.csv must contain only numeric columns (no raw strings). Error: {e}")
    st.stop()

if X.shape[0] != len(df_clean):
    st.warning(
        f"Row count mismatch: encode_data.csv has {X.shape[0]} rows but cleaned_data.csv has {len(df_clean)} rows.\n"
        "This app assumes they match exactly by row index."
    )

# Input
with st.form("pref_form"):
    # City: single select from list
    sel_city = st.selectbox(
        "Select City",
        options=[""] + city_area_main_options,
        index=0
    )

    # Cuisine: multiselect of raw cuisine
    sel_cuisines_raw = st.multiselect(
        "Select Cuisine",
        options=cuisine_raw_options,
        default=[]
    )

    # Rating input (slider)
    if rating_options:
       user_rating  = st.selectbox("Choose Rating ", options=[""] + [str(r) for r in rating_options], index=0)
    
    # Cost Input
    if cost_options:
       user_cost  = st.selectbox("Choose Cost ", options=[""] + [str(r) for r in cost_options], index=0)   

    submitted = st.form_submit_button("Get recommendations")

if not submitted:
    st.info("Select options and press 'Get recommendations'.")
    st.stop()

# Resolve final inputs
user_city_raw = sel_city.strip() if isinstance(sel_city, str) else ""

selected_raws = sel_cuisines_raw if isinstance(sel_cuisines_raw, list) else []

# Convert selected raw cuisine strings to token list for ML-Binarizer
user_cuisines_list = parse_raw_cuisine_strings(selected_raws)

user_rating_val = float(user_rating)
user_cost_val = float(user_cost)

# Encode user input
try:
    area, main_city = extract_city_parts(user_city_raw)
    if main_city == "" or main_city is None:
        main_city = "0"

    # Scaling
    num = np.array([[float(user_rating_val), float(user_cost_val)]])
    num_scaled_u = scaler.transform(num)

    # ohe for city/main_city
    city_df_u = pd.DataFrame([[area, main_city]], columns=['city','main_city'])
    city_ohe_u = ohe.transform(city_df_u)
    if hasattr(city_ohe_u, "toarray"):
        city_ohe_u = city_ohe_u.toarray()

    # mlb for cuisines
    cuisine_vec_u = mlb.transform([user_cuisines_list])
    if hasattr(cuisine_vec_u, "toarray"):
        cuisine_vec_u = cuisine_vec_u.toarray()

    # final user vector
    user_vec = np.hstack([num_scaled_u, city_ohe_u, cuisine_vec_u]).astype(np.float32)
except Exception as e:
    st.error("Error encoding user input:")
    st.exception(e)
    st.stop()

# Dimension check
if user_vec.shape[1] != X.shape[1]:
    st.error(
        f"Feature dimension mismatch: user vector has {user_vec.shape[1]} features but encoded dataset has {X.shape[1]}.\n\n"
        "Make sure encode_data.csv was produced by the same pipeline (scaler, ohe, mlb) and columns are in the same order."
    )
    st.stop()

#  Compute similarity & recommend
try:
    sims = cosine_similarity(user_vec, X).flatten()
except Exception as e:
    st.error("Error computing cosine similarity:")
    st.exception(e)
    st.stop()

# Shows Top 10 Matches Restaurents
top_k = 10
top_idx = np.argsort(-sims)[:top_k]

results_rows = []
for rank, idx in enumerate(top_idx, start=1):
    if idx < 0 or idx >= len(df_clean):
        continue
    rec = {
        "rank": rank,
        "encoded_index": int(idx),
        "Similarity": float(sims[idx])
    }
    row = df_clean.iloc[int(idx)]
    rec.update({
        "Name": row.get("name", ""),
        "City": row.get("city", ""),
        "Cuisine": row.get("cuisine", ""),
        "Cost": row.get("cost", ""),
        "Rating": row.get("rating", ""),
        "Rating_count": row.get("rating_count", ""),
        "Address": row.get("address", ""),
    })
    results_rows.append(rec)

st.success(f"Top {len(results_rows)} recommendations")

df_results = pd.DataFrame(results_rows)

# Display columns
display_cols = ["Name", "City", "Cuisine", "Cost", "Rating", "Rating_count", "Address", "Similarity"]

# # Reindex to ensure columns exist
df_display = df_results.reindex(columns=display_cols)

# Format cost as ₹ value
df_display["Cost"] = df_display["Cost"].apply(lambda x: f"₹ {x}")

st.dataframe(df_display, hide_index=True)