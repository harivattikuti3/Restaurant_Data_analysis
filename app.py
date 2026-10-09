# ==============================================================================
# UNIFIED RESTAURANT ANALYTICS & MACHINE LEARNING SUITE
# ==============================================================================
# Description:
#   A comprehensive Streamlit multi-section application combining 4 key ML modules:
#   1. Restaurant Rating Predictor (Linear Regression Pipeline)
#   2. Restaurant Recommendation Engine (Cosine Similarity Content-Based Filtering)
#   3. Cuisine Classifier (Random Forest & Logistic Regression)
#   4. Location-Based Geospatial Analysis (Folium Mapping & Spatial Insights)
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import streamlit as st
import streamlit.components.v1 as components

# Scikit-Learn Preprocessing & Pipeline Modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Scikit-Learn Models & Evaluation Metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------------
# 1. STREAMLIT GLOBAL PAGE CONFIGURATION & THEME STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Data analysis",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS for Navigation, Cards, and Headers
st.markdown("""
    <style>
    .suite-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4EC8B0, #4EA8DE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .suite-sub {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1d3557;
        margin-bottom: 0.5rem;
    }
    .card-banner {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .rec-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        border-left: 6px solid #FF6B6B;
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
# 2. SHARED DATA LOADING & CACHING FUNCTION
# ------------------------------------------------------------------------------
@st.cache_data
def load_master_dataset():
    """
    Loads raw dataset, cleans column whitespace, drops invalid rows, 
    and returns a clean master pandas DataFrame.
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
        return None

    # Strip column header spaces
    df_raw.columns = df_raw.columns.str.strip()
    return df_raw

# Load Dataset
df_raw = load_master_dataset()

# Handle missing dataset file
if df_raw is None:
    st.error("⚠️ Dataset file ('Dataset .csv') not found in project directory.")
    uploaded_file = st.sidebar.file_uploader("Upload Restaurant Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()
    else:
        st.stop()

# Master dataset copy
df = df_raw.copy()

# ------------------------------------------------------------------------------
# 3. CACHED MODEL BUILDERS
# ------------------------------------------------------------------------------

# A. RATING PREDICTION MODEL (Linear Regression Pipeline)
@st.cache_resource
def build_rating_model(df):
    num_features = ['Votes', 'Price range', 'Average Cost for two']
    cat_features = ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 'Rating color', 'Rating text']
    all_features = cat_features + num_features
    target_col = 'Aggregate rating'

    X = df[all_features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)])
    model = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    regressor = model.named_steps['regressor']
    prep = model.named_steps['preprocessor']
    feature_names = prep.get_feature_names_out()
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': regressor.coef_}).sort_values(by='Coefficient', ascending=False)

    return model, X_test, y_test, y_pred, mse, r2, coef_df

# B. RECOMMENDATION PREPROCESSOR & MATRIX
@st.cache_resource
def build_recommendation_matrix(df):
    req_columns = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'Address', 'Locality', 'City', 'Locality Verbose', 'Currency', 'Has Table booking', 'Has Online delivery', 'Cuisines', 'Price range', 'Aggregate rating', 'Votes', 'Rating color', 'Rating text', 'Average Cost for two']
    avail_cols = [c for c in req_columns if c in df.columns]
    rec_df = df[avail_cols].copy()

    drop_info = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'Address', 'Locality', 'City', 'Locality Verbose', 'Currency', 'Has Table booking', 'Has Online delivery']
    feature_cols = [c for c in avail_cols if c not in drop_info]
    feat_df = rec_df[feature_cols].copy()

    num_cols = [c for c in ['Aggregate rating', 'Votes', 'Average Cost for two', 'Price range'] if c in feat_df.columns]
    cat_cols = [c for c in ['Cuisines', 'Rating color', 'Rating text'] if c in feat_df.columns]

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    rec_preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    X_processed = rec_preprocessor.fit_transform(feat_df)

    return rec_df, feat_df, rec_preprocessor, X_processed

# C. CUISINE CLASSIFICATION MODELS
@st.cache_resource
def build_cuisine_models(df):
    cdf = df.copy()
    cdf['Cuisines'] = cdf['Cuisines'].astype(str).str.split(',').str[0].str.strip()
    cdf = cdf.dropna(subset=['Cuisines'])
    cdf = cdf[cdf['Cuisines'] != 'nan']

    X = cdf[['Country Code', 'City', 'Locality', 'Locality Verbose']].copy()
    y = cdf['Cuisines'].copy()

    num_cols = ['Country Code']
    cat_cols = ['City', 'Locality', 'Locality Verbose']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Logistic Regression
    lr_model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred, average='weighted', zero_division=0)
    lr_rec = recall_score(y_test, lr_pred, average='weighted', zero_division=0)
    lr_f1 = f1_score(y_test, lr_pred, average='weighted', zero_division=0)

    # 2. Random Forest Classifier
    rf_model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1))])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)

    metrics_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest Classifier'],
        'Accuracy': [lr_acc, rf_acc],
        'Precision': [lr_prec, rf_prec],
        'Recall': [lr_rec, rf_rec],
        'F1-Score': [lr_f1, rf_f1]
    })

    return cdf, lr_model, rf_model, metrics_df, X_test, y_test

# Initialize cached artifacts
rating_model, rating_X_test, rating_y_test, rating_y_pred, rating_mse, rating_r2, rating_coef_df = build_rating_model(df)
rec_df, rec_feat_df, rec_preprocessor, rec_X_processed = build_recommendation_matrix(df)
cdf, lr_model, rf_model, cls_metrics_df, cls_X_test, cls_y_test = build_cuisine_models(df)

# ------------------------------------------------------------------------------
# 4. MAIN NAVIGATION SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.markdown("# 🍽️ Navigation Menu")
selected_section = st.sidebar.radio(
    "Select Module Section:",
    [
        "🔮 Section 1: Rating Predictor",
        "🍴 Section 2: Recommendation Engine",
        "🍜 Section 3: Cuisine Classification",
        "🗺️ Section 4: Location & Geospatial Map"
    ],
    index=0
)

# Header Title Banner
st.markdown('<div class="suite-header">🍽️ Restaurant Machine Learning & Analytics Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified Intelligence Platform: Rating Prediction, Content-Based Recommendation, Cuisine Classification & Geospatial Mapping</div>', unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: RESTAURANT RATING PREDICTOR
# ==============================================================================
if "Section 1" in selected_section:
    st.markdown('<div class="section-title">🔮 Section 1: Restaurant Rating Predictor</div>', unsafe_allow_html=True)
    st.write("Predict aggregate restaurant ratings using Machine Learning (Linear Regression with Scikit-Learn Pipeline).")

    # Sidebar Controls
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Rating Predictor Inputs")

    unique_cities = sorted(df['City'].dropna().unique().tolist())
    unique_cuisines = sorted(df['Cuisines'].dropna().unique().tolist())
    unique_colors = df['Rating color'].dropna().unique().tolist()
    unique_texts = df['Rating text'].dropna().unique().tolist()

    default_city = 'Makati City' if 'Makati City' in unique_cities else unique_cities[0]
    sel_city = st.sidebar.selectbox("City", unique_cities, index=unique_cities.index(default_city), key="r_city")
    sel_cuisines = st.sidebar.selectbox("Cuisines", unique_cuisines, index=0, key="r_cuisines")
    sel_table = st.sidebar.radio("Has Table Booking?", ['Yes', 'No'], index=0, key="r_table")
    sel_delivery = st.sidebar.radio("Has Online Delivery?", ['No', 'Yes'], index=0, key="r_delivery")
    sel_price = st.sidebar.select_slider("Price Range Tier (1-4)", options=[1, 2, 3, 4], value=3, key="r_price")
    sel_votes = st.sidebar.number_input("Total Votes", min_value=0, max_value=50000, value=314, step=10, key="r_votes")
    sel_cost = st.sidebar.number_input("Average Cost for Two", min_value=0, max_value=500000, value=600, step=50, key="r_cost")
    sel_color = st.sidebar.selectbox("Rating Color Descriptor", unique_colors, index=0, key="r_color")
    sel_text = st.sidebar.selectbox("Rating Text Descriptor", unique_texts, index=0, key="r_text")

    user_rating_df = pd.DataFrame([{
        'City': sel_city, 'Cuisines': sel_cuisines, 'Has Table booking': sel_table,
        'Has Online delivery': sel_delivery, 'Price range': sel_price, 'Votes': sel_votes,
        'Average Cost for two': sel_cost, 'Rating color': sel_color, 'Rating text': sel_text
    }])

    # Dashboard Tabs inside Section 1
    t1, t2, t3, t4 = st.tabs(["🔮 Single Prediction", "📊 Model Performance", "📈 Feature Importance", "📁 Dataset Explorer"])

    with t1:
        st.subheader("🎯 Rating Prediction Output")
        st.write("Input Configuration:")
        st.dataframe(user_rating_df, use_container_width=True)

        raw_pred = rating_model.predict(user_rating_df)[0]
        bounded_pred = round(max(0.0, min(5.0, raw_pred)), 2)

        st.markdown(f"""
            <div class="card-banner">
                <h3 style="margin:0; font-size:1.3rem;">Predicted Aggregate Rating</h3>
                <div style="font-size: 3.5rem; font-weight: 800; margin: 4px 0;">⭐ {bounded_pred} / 5.0</div>
                <p style="margin:0; opacity:0.9;">Raw Model Output: {raw_pred:.4f}</p>
            </div>
        """, unsafe_allow_html=True)

    with t2:
        st.subheader("📊 Regression Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R-squared (R²)", f"{rating_r2:.4f}")
        c2.metric("Mean Squared Error (MSE)", f"{rating_mse:.4f}")
        c3.metric("RMSE", f"{np.sqrt(rating_mse):.4f}")
        c4.metric("Dataset Size", f"{len(df):,} records")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(rating_y_test, rating_y_pred, color='#2ec4b6', alpha=0.5)
        min_v = min(rating_y_test.min(), rating_y_pred.min())
        max_v = max(rating_y_test.max(), rating_y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], color='#e71d36', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual Ratings')
        ax.set_ylabel('Predicted Ratings')
        ax.set_title('Actual vs. Predicted Ratings Scatter Plot')
        ax.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig)

    with t3:
        st.subheader("📈 Linear Regression Feature Coefficients")
        st.dataframe(rating_coef_df.reset_index(drop=True), height=350, use_container_width=True)

    with t4:
        st.subheader("📁 Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

# ==============================================================================
# SECTION 2: RESTAURANT RECOMMENDATION ENGINE
# ==============================================================================
elif "Section 2" in selected_section:
    st.markdown('<div class="section-title">🍴 Section 2: Restaurant Recommendation Engine</div>', unsafe_allow_html=True)
    st.write("Content-Based Filtering Recommendation Engine using Cosine Similarity on restaurant feature vectors.")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Recommendation Preferences")

    unique_cuisines = sorted(df['Cuisines'].dropna().unique().tolist())
    unique_colors = df['Rating color'].dropna().unique().tolist()
    unique_texts = df['Rating text'].dropna().unique().tolist()

    rec_cuisine = st.sidebar.selectbox("Preferred Cuisine", unique_cuisines, index=0, key="rec_cuisine")
    rec_price = st.sidebar.select_slider("Price Range Tier", options=[1, 2, 3, 4], value=3, key="rec_price")
    rec_rating = st.sidebar.slider("Preferred Rating (0.0 - 5.0)", 0.0, 5.0, 4.5, 0.1, key="rec_rating")
    rec_cost = st.sidebar.number_input("Average Cost for Two", 0, 500000, 600, 50, key="rec_cost")
    rec_votes = st.sidebar.number_input("Preferred Minimum Votes", 0, 50000, 300, 25, key="rec_votes")
    rec_color = st.sidebar.selectbox("Rating Color Preference", unique_colors, index=0, key="rec_color")
    rec_text = st.sidebar.selectbox("Rating Text Preference", unique_texts, index=0, key="rec_text")
    rec_N = st.sidebar.slider("Number of Recommendations (N)", 1, 20, 5, key="rec_N")

    user_rec_df = pd.DataFrame([{
        'Cuisines': rec_cuisine, 'Price range': rec_price, 'Aggregate rating': rec_rating,
        'Average Cost for two': rec_cost, 'Votes': rec_votes, 'Rating color': rec_color, 'Rating text': rec_text
    }])

    # Compute Recommendations via Cosine Similarity
    user_processed = rec_preprocessor.transform(user_rec_df)
    sim_scores = cosine_similarity(user_processed, rec_X_processed)[0]
    top_indices = np.argsort(sim_scores)[::-1][:rec_N]

    recommendations = rec_df.iloc[top_indices].copy()
    recommendations['Similarity Score'] = sim_scores[top_indices]
    recommendations['Match Percentage'] = [f"{round(s * 100, 1)}%" for s in sim_scores[top_indices]]

    rt1, rt2, rt3 = st.tabs(["🏆 Top Recommendations", "📊 Similarity Analytics", "📁 Recommendation Data"])

    with rt1:
        st.subheader(f"✨ Top {rec_N} Matching Restaurants for You")
        for idx, row in recommendations.reset_index(drop=True).iterrows():
            st.markdown(f"""
                <div class="rec-card">
                    <span class="similarity-badge">Match: {row['Match Percentage']}</span>
                    <div style="font-size:1.3rem; font-weight:700; color:#1d3557;">#{idx+1} {row.get('Restaurant Name', 'Restaurant')}</div>
                    <p style="color:#457b9d; margin-bottom:6px; font-weight:500;">📍 {row.get('City', 'N/A')} | {row.get('Address', '')}</p>
                    <div>
                        <span class="detail-tag">🍲 {row.get('Cuisines', 'N/A')}</span>
                        <span class="detail-tag">⭐ {row.get('Aggregate rating', 'N/A')} / 5.0 ({row.get('Votes', 0)} votes)</span>
                        <span class="detail-tag">💰 Cost for 2: {row.get('Currency', '$')} {row.get('Average Cost for two', 'N/A')}</span>
                        <span class="detail-tag">🏷️ Price Tier: {row.get('Price range', 'N/A')}/4</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with rt2:
        st.subheader("📊 Similarity Score Comparison")
        fig_sim, ax_sim = plt.subplots(figsize=(7, 4))
        ax_sim.barh(recommendations['Restaurant Name'], recommendations['Similarity Score'], color='#2a9d8f')
        ax_sim.set_xlabel('Cosine Similarity Score')
        ax_sim.set_title('Top Restaurant Similarity Scores')
        ax_sim.invert_yaxis()
        ax_sim.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_sim)

    with rt3:
        st.subheader("📁 Recommendation Matrix Data")
        st.dataframe(recommendations, use_container_width=True)

# ==============================================================================
# SECTION 3: CUISINE CLASSIFICATION
# ==============================================================================
elif "Section 3" in selected_section:
    st.markdown('<div class="section-title">🍜 Section 3: Cuisine Classification</div>', unsafe_allow_html=True)
    st.write("Predict primary restaurant cuisines based on location parameters using Machine Learning classifiers.")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Classifier Inputs")

    model_choice = st.sidebar.radio("Choose Model:", ["Random Forest Classifier (Recommended)", "Logistic Regression"], index=0, key="c_model")
    active_cls_model = rf_model if "Random Forest" in model_choice else lr_model

    unique_country_codes = sorted(df['Country Code'].dropna().unique().tolist())
    unique_cities = sorted(df['City'].dropna().unique().tolist())
    unique_localities = sorted(df['Locality'].dropna().unique().tolist())
    unique_locality_verboses = sorted(df['Locality Verbose'].dropna().unique().tolist())

    c_country = st.sidebar.selectbox("Country Code", unique_country_codes, index=0, key="c_country")
    c_city = st.sidebar.selectbox("City", unique_cities, index=0, key="c_city")
    
    city_localities = sorted(df[df['City'] == c_city]['Locality'].dropna().unique().tolist())
    loc_opts = city_localities if len(city_localities) > 0 else unique_localities
    c_locality = st.sidebar.selectbox("Locality", loc_opts, index=0, key="c_locality")

    loc_verboses = sorted(df[df['Locality'] == c_locality]['Locality Verbose'].dropna().unique().tolist())
    verb_opts = loc_verboses if len(loc_verboses) > 0 else unique_locality_verboses
    c_verbose = st.sidebar.selectbox("Locality Verbose", verb_opts, index=0, key="c_verbose")

    user_cls_df = pd.DataFrame([{
        'Country Code': c_country, 'City': c_city, 'Locality': c_locality, 'Locality Verbose': c_verbose
    }])

    ct1, ct2, ct3 = st.tabs(["🔮 Single Classification", "📊 Model Comparison", "📈 Cuisine Distribution"])

    with ct1:
        st.subheader("🎯 Primary Cuisine Prediction")
        st.write("Selected Location Input:")
        st.dataframe(user_cls_df, use_container_width=True)

        predicted_cuisine = active_cls_model.predict(user_cls_df)[0]

        st.markdown(f"""
            <div class="card-banner" style="background: linear-gradient(135deg, #4EA8DE, #7400B8);">
                <h3 style="margin:0; font-size:1.3rem;">Predicted Primary Cuisine</h3>
                <div style="font-size: 3.2rem; font-weight: 800; margin: 4px 0;">🍲 {predicted_cuisine}</div>
                <p style="margin:0; opacity:0.9;">Model Used: {model_choice}</p>
            </div>
        """, unsafe_allow_html=True)

    with ct2:
        st.subheader("📊 Classifier Performance Comparison")
        st.dataframe(cls_metrics_df, use_container_width=True)

        fig_m, ax_m = plt.subplots(figsize=(6, 4))
        x_idx = np.arange(len(cls_metrics_df))
        w = 0.35
        ax_m.bar(x_idx - w/2, cls_metrics_df['Accuracy'], w, label='Accuracy', color='#4EA8DE')
        ax_m.bar(x_idx + w/2, cls_metrics_df['F1-Score'], w, label='F1-Score', color='#7400B8')
        ax_m.set_xticks(x_idx)
        ax_m.set_xticklabels(cls_metrics_df['Model'])
        ax_m.set_ylabel('Score')
        ax_m.set_ylim(0, 1.0)
        ax_m.legend()
        ax_m.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_m)

    with ct3:
        st.subheader("📈 Top Cuisines Distribution in Dataset")
        top_n = st.slider("Select Top N Cuisines", 5, 30, 15, key="c_top_n")
        counts = cdf['Cuisines'].value_counts().head(top_n)

        fig_c, ax_c = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x=counts.values, y=counts.index, palette='crest', ax=ax_c)
        ax_c.set_xlabel('Number of Restaurants')
        ax_c.set_ylabel('Cuisine')
        ax_c.set_title(f'Top {top_n} Primary Cuisines')
        ax_c.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_c)

# ==============================================================================
# SECTION 4: LOCATION & GEOSPATIAL MAP
# ==============================================================================
elif "Section 4" in selected_section:
    st.markdown('<div class="section-title">🗺️ Section 4: Location & Geospatial Analysis</div>', unsafe_allow_html=True)
    st.write("Spatial mapping, geographic density metrics, price-rating correlation, and hidden gem discovery.")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Spatial Analysis Filters")

    map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    map_df['Latitude'] = pd.to_numeric(map_df['Latitude'], errors='coerce')
    map_df['Longitude'] = pd.to_numeric(map_df['Longitude'], errors='coerce')
    map_df = map_df.dropna(subset=['Latitude', 'Longitude'])
    map_df = map_df[(map_df['Latitude'] != 0) & (map_df['Longitude'] != 0)]

    map_cities = ['All Cities'] + sorted(map_df['City'].dropna().unique().tolist())
    map_sel_city = st.sidebar.selectbox("Filter City", map_cities, index=0, key="m_city")
    map_price_opts = sorted(map_df['Price range'].dropna().unique().tolist())
    map_sel_prices = st.sidebar.multiselect("Price Range Tier", map_price_opts, default=map_price_opts, key="m_prices")
    map_min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5, key="m_rating")
    max_map_markers = st.sidebar.slider("Maximum Map Markers", 100, 5000, 1000, 100, key="m_max")

    filt_map_df = map_df.copy()
    if map_sel_city != 'All Cities':
        filt_map_df = filt_map_df[filt_map_df['City'] == map_sel_city]
    if map_sel_prices:
        filt_map_df = filt_map_df[filt_map_df['Price range'].isin(map_sel_prices)]
    filt_map_df = filt_map_df[filt_map_df['Aggregate rating'] >= map_min_rating]

    lt1, lt2, lt3, lt4 = st.tabs(["🗺️ Interactive Folium Map", "🏙️ Concentration Analysis", "📊 Price vs Rating", "💎 Potential Hidden Gems"])

    with lt1:
        st.subheader("📍 Interactive Geospatial Map")
        st.write(f"Displaying **{min(len(filt_map_df), max_map_markers)}** clustered restaurant markers:")

        if len(filt_map_df) == 0:
            st.warning("No restaurants match active filter criteria.")
        else:
            map_center = [filt_map_df['Latitude'].mean(), filt_map_df['Longitude'].mean()]
            fmap = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap")
            marker_cluster = MarkerCluster().add_to(fmap)

            for index, row in filt_map_df.head(max_map_markers).iterrows():
                popup_html = f"""
                    <div style="font-family: sans-serif; min-width: 170px;">
                        <h4 style="margin:0 0 4px 0; color:#1d3557;">{row['Restaurant Name']}</h4>
                        <b>City:</b> {row['City']}<br>
                        <b>Rating:</b> ⭐ {row['Aggregate rating']} / 5.0<br>
                        <b>Cuisines:</b> {row['Cuisines']}<br>
                        <b>Cost for 2:</b> {row.get('Currency', '$')} {row.get('Average Cost for two', 'N/A')}
                    </div>
                """
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_html, max_width=240),
                    tooltip=row['Restaurant Name']
                ).add_to(marker_cluster)

            components.html(fmap._repr_html_(), height=550)

    with lt2:
        st.subheader("🏙️ Restaurant Concentration by Location")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("#### Top 15 Cities")
            city_conc = filt_map_df['City'].value_counts().head(15).reset_index()
            city_conc.columns = ['City', 'Count']
            fig_conc, ax_conc = plt.subplots(figsize=(6, 4))
            sns.barplot(data=city_conc, x='Count', y='City', palette='viridis', ax=ax_conc)
            ax_conc.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig_conc)
        with col_c2:
            st.write("#### Top 10 Localities")
            loc_conc = filt_map_df['Locality'].value_counts().head(10).reset_index()
            loc_conc.columns = ['Locality', 'Count']
            fig_lconc, ax_lconc = plt.subplots(figsize=(6, 4))
            sns.barplot(data=loc_conc, x='Count', y='Locality', palette='magma', ax=ax_lconc)
            ax_lconc.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig_lconc)

    with lt3:
        st.subheader("📊 Price Range vs Average Rating by City")
        city_stats = filt_map_df.groupby('City').agg({'Aggregate rating': 'mean', 'Price range': 'mean', 'Restaurant ID': 'count'}).reset_index()
        city_stats.columns = ['City', 'Avg Rating', 'Avg Price Range', 'Count']
        fig_sc, ax_sc = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(data=city_stats, x='Avg Price Range', y='Avg Rating', size='Count', hue='City', sizes=(100, 800), legend=False, ax=ax_sc)
        ax_sc.set_title('Average Rating vs Price Range Tier')
        ax_sc.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_sc)

    with lt4:
        st.subheader("💎 Potential Hidden Gem Localities")
        st.write("Localities with **Average Rating > 4.0** and **Average Votes < 100**:")
        gems = filt_map_df.groupby('Locality').agg({'Aggregate rating': 'mean', 'Votes': 'mean', 'Restaurant ID': 'count'}).reset_index()
        gems = gems[(gems['Aggregate rating'] > 4.0) & (gems['Votes'] < 100)].sort_values(by='Aggregate rating', ascending=False)
        st.dataframe(gems.rename(columns={'Restaurant ID': 'Restaurant Count'}), use_container_width=True)

# ------------------------------------------------------------------------------
# 5. SCRIPT ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
