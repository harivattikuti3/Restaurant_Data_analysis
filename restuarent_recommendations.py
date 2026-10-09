# ==============================================================================
# RESTAURANT RECOMMENDATION SYSTEM & STREAMLIT WEB APP
# ==============================================================================
# Description:
#   This application builds a Content-Based Filtering Recommendation Engine 
#   using Cosine Similarity on preprocessed numerical and categorical restaurant 
#   features (Cuisines, Price Range, Aggregate Rating, Cost for Two, Votes, etc.).
#   It offers an interactive Streamlit dashboard allowing users to input their 
#   dining preferences and receive top N matching restaurant recommendations.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Scikit-Learn Preprocessing & Similarity Metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Recommender System",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for recommendation cards and UI styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #E63946, #F4A261);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .rec-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid #E63946;
        transition: transform 0.2s ease-in-out;
    }
    .rec-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1d3557;
        margin-bottom: 4px;
    }
    .similarity-badge {
        background-color: #2a9d8f;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        float: right;
    }
    .detail-tag {
        background-color: #f1faee;
        color: #1d3557;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 6px;
        display: inline-block;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING (CACHED)
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    """
    Loads dataset from local file path, strips column whitespace,
    and returns full and required data subsets.
    """
    candidate_paths = [
        os.path.join('dataset', 'Dataset .csv'),
        'Dataset .csv',
        os.path.join('dataset', 'Dataset.csv'),
        'Dataset.csv'
    ]
    
    df_raw = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df_raw = pd.read_csv(path)
                break
            except Exception:
                continue
                
    if df_raw is None:
        return None, None, None

    # Clean whitespace from column headers
    df_raw.columns = df_raw.columns.str.strip()
    
    req_columns = [
        'Restaurant ID', 'Restaurant Name', 'Country Code', 'Address', 
        'Locality', 'City', 'Locality Verbose', 'Currency', 
        'Has Table booking', 'Has Online delivery', 'Cuisines', 
        'Price range', 'Aggregate rating', 'Votes', 
        'Rating color', 'Rating text', 'Average Cost for two'
    ]
    
    # Filter dataset to existing columns
    available_req = [col for col in req_columns if col in df_raw.columns]
    df = df_raw[available_req].copy()

    # Feature dataframe used for vector similarity computation
    drop_info_cols = [
        'Restaurant ID', 'Restaurant Name', 'Country Code', 'Address', 
        'Locality', 'City', 'Locality Verbose', 'Currency', 
        'Has Table booking', 'Has Online delivery'
    ]
    feature_cols = [col for col in available_req if col not in drop_info_cols]
    feature_df = df[feature_cols].copy()

    return df, feature_df, df_raw

@st.cache_resource
def build_preprocessor_and_matrix(feature_df):
    """
    Constructs Scikit-Learn ColumnTransformer for numerical scaling and 
    categorical one-hot encoding, then transforms feature_df into matrix X_processed.
    """
    num_data = ['Aggregate rating', 'Votes', 'Average Cost for two', 'Price range']
    cat_data = ['Cuisines', 'Rating color', 'Rating text']

    # Filter features based on available columns
    num_cols = [c for c in num_data if c in feature_df.columns]
    cat_cols = [c for c in cat_data if c in feature_df.columns]

    # Preprocessing Pipeline for Numerical Features
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing Pipeline for Categorical Features
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combined ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Fit transformer and generate processed matrix
    X_processed = preprocessor.fit_transform(feature_df)
    
    return preprocessor, X_processed, num_cols, cat_cols

# ------------------------------------------------------------------------------
# 3. RECOMMENDATION ENGINE FUNCTION
# ------------------------------------------------------------------------------
def recommend_restaurants(user_input_df, preprocessor, X_processed, df, N=5):
    """
    Computes Cosine Similarity between preprocessed user preferences 
    and all restaurant feature vectors in the dataset.
    
    Returns:
        pd.DataFrame: Top N recommended restaurants with similarity scores.
    """
    # Vectorize user preferences using fitted preprocessor
    user_processed = preprocessor.transform(user_input_df)
    
    # Calculate cosine similarity scores against all dataset items
    similarity_matrix = cosine_similarity(user_processed, X_processed)
    
    # Sort similarity scores in descending order
    similarity_scores = list(enumerate(similarity_matrix[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Extract top N restaurant indices
    top_indices = [i[0] for i in similarity_scores[:N]]
    top_scores = [i[1] for i in similarity_scores[:N]]

    # Extract matching restaurant records from original dataframe
    result = df.iloc[top_indices].copy()
    result['Similarity Score'] = top_scores
    result['Match Percentage'] = [f"{round(score * 100, 1)}%" for score in top_scores]
    
    return result

# Load Data
df, feature_df, df_raw = load_and_clean_data()

# Handle missing dataset case
if df is None:
    st.error("⚠️ Dataset ('Dataset .csv') not found in project folder.")
    uploaded_file = st.sidebar.file_uploader("Upload Restaurant Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()
        df = df_raw.copy()
        feature_cols = ['Cuisines', 'Price range', 'Aggregate rating', 'Votes', 'Rating color', 'Rating text', 'Average Cost for two']
        feature_df = df[[c for c in feature_cols if c in df.columns]]
    else:
        st.stop()

# Build preprocessor and processed feature matrix
preprocessor, X_processed, num_cols, cat_cols = build_preprocessor_and_matrix(feature_df)

# ------------------------------------------------------------------------------
# 4. STREAMLIT INTERFACE & SIDEBAR PREFERENCE FORM
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🍴 Restaurant Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Discover top-rated restaurants tailored to your cuisine, budget, and popularity preferences using Cosine Similarity Content-Based Filtering.</div>', unsafe_allow_html=True)

# Sidebar Form Controls for User Input
st.sidebar.header("🎯 Your Dining Preferences")

# Extract unique values for dynamic selectboxes
unique_cuisines = sorted(df['Cuisines'].dropna().unique().tolist()) if 'Cuisines' in df else ['Italian']
unique_colors = df['Rating color'].dropna().unique().tolist() if 'Rating color' in df else ['Dark Green']
unique_texts = df['Rating text'].dropna().unique().tolist() if 'Rating text' in df else ['Excellent']

selected_cuisine = st.sidebar.selectbox("Preferred Cuisine", unique_cuisines, index=0)
selected_price_range = st.sidebar.select_slider("Price Range Tier (1: Affordable, 4: Fine Dining)", options=[1, 2, 3, 4], value=3)
selected_rating = st.sidebar.slider("Preferred Aggregate Rating (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
selected_cost = st.sidebar.number_input("Average Cost for Two", min_value=0, max_value=500000, value=600, step=50)
selected_votes = st.sidebar.number_input("Preferred Popularity (Minimum Votes)", min_value=0, max_value=50000, value=300, step=25)
selected_color = st.sidebar.selectbox("Rating Color Preference", unique_colors, index=0)
selected_text = st.sidebar.selectbox("Rating Text Preference", unique_texts, index=0)

num_recommendations = st.sidebar.slider("Number of Recommendations (N)", min_value=1, max_value=20, value=5)

# Construct user input DataFrame matching pipeline feature columns
user_input_df = pd.DataFrame([{
    'Cuisines': selected_cuisine,
    'Price range': selected_price_range,
    'Aggregate rating': selected_rating,
    'Average Cost for two': selected_cost,
    'Votes': selected_votes,
    'Rating color': selected_color,
    'Rating text': selected_text
}])

# Trigger recommendations calculation
recommendations = recommend_restaurants(user_input_df, preprocessor, X_processed, df, N=num_recommendations)

# ------------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT & DISPLAY TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏆 Top Recommendations", 
    "📊 Similarity Analytics", 
    "📁 Dataset Explorer"
])

# ------------------------------------------------------------------------------
# TAB 1: TOP RECOMMENDATIONS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"✨ Top {num_recommendations} Restaurant Matches for You")
    
    for idx, row in recommendations.reset_index(drop=True).iterrows():
        match_pct = row['Match Percentage']
        name = row.get('Restaurant Name', 'Restaurant')
        city = row.get('City', 'N/A')
        cuisines = row.get('Cuisines', 'N/A')
        rating = row.get('Aggregate rating', 'N/A')
        cost = row.get('Average Cost for two', 'N/A')
        currency = row.get('Currency', '$')
        votes = row.get('Votes', '0')
        price_tier = row.get('Price range', 'N/A')
        address = row.get('Address', '')
        booking = row.get('Has Table booking', 'No')
        delivery = row.get('Has Online delivery', 'No')

        st.markdown(f"""
            <div class="rec-card">
                <span class="similarity-badge">Match: {match_pct}</span>
                <div class="rec-title">#{idx+1} {name}</div>
                <p style="color: #457b9d; margin-bottom: 8px; font-weight: 500;">📍 {city} | {address}</p>
                <div>
                    <span class="detail-tag">🍲 {cuisines}</span>
                    <span class="detail-tag">⭐ {rating} / 5.0 ({votes} votes)</span>
                    <span class="detail-tag">💰 Cost for 2: {currency} {cost}</span>
                    <span class="detail-tag">🏷️ Price Tier: {price_tier}/4</span>
                    <span class="detail-tag">🪑 Table Booking: {booking}</span>
                    <span class="detail-tag">🛵 Online Delivery: {delivery}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.write("#### 📋 Detailed Recommendation Matrix")
    st.dataframe(recommendations, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: SIMILARITY ANALYTICS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Recommendation Similarity Analysis")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.write("#### Similarity Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(recommendations['Restaurant Name'], recommendations['Similarity Score'], color='#E63946')
        ax.set_xlabel('Cosine Similarity Score')
        ax.set_title('Top Restaurant Match Scores')
        ax.invert_yaxis()
        ax.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig)

    with col_chart2:
        st.write("#### Rating vs Price Tier of Recommended Places")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(recommendations['Price range'], recommendations['Aggregate rating'], color='#2a9d8f', s=120)
        ax2.set_xlabel('Price Range Tier (1-4)')
        ax2.set_ylabel('Aggregate Rating')
        ax2.set_title('Price Tier vs Rating Distribution')
        ax2.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig2)

# ------------------------------------------------------------------------------
# TAB 3: DATASET EXPLORER
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📁 Full Restaurant Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write(f"Total Dataset Size: **{len(df):,} restaurants**")

# ------------------------------------------------------------------------------
# 6. DIRECT SCRIPT EXECUTION SUPPORT (CLI BACKWARD COMPATIBILITY)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Content-Based Restaurant Recommendation Engine Loaded ---")
    print(f"Dataset Size: {len(df)} restaurants across multiple cities.")
