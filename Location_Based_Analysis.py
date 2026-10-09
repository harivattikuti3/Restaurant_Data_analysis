# ==============================================================================
# LOCATION-BASED RESTAURANT GEOGRAPHIC ANALYSIS & STREAMLIT MAP DASHBOARD
# ==============================================================================
# Description:
#   This application performs Spatial Data Analytics and Interactive Mapping on 
#   restaurant location coordinates (Latitude & Longitude). It visualizes 
#   restaurant density clusters via Folium, analyzes market concentration by 
#   City & Locality, investigates Price Range vs. Rating correlations, detects 
#   Online Delivery adoption, and discovers "Hidden Gems" (high rating, low vote count).
# ==============================================================================

import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Location-Based Analysis App",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design and metric badges
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2A9D8F, #E76F51);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .gem-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #2A9D8F;
    }
    .gem-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #264653;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. DATA LOADING & GEOGRAPHIC CLEANING (CACHED)
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_location_data():
    """
    Loads raw dataset, cleans column whitespace, drops rows missing 
    Latitude or Longitude coordinates, and returns processed DataFrames.
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

    # Drop rows missing valid coordinates
    df = df_raw.dropna(subset=['Latitude', 'Longitude']).copy()
    
    # Ensure numeric coordinates
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Exclude zero/invalid coordinates if any
    df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]

    return df

# Load cleaned data
df = load_and_clean_location_data()

# Fallback for missing file
if df is None:
    st.error("⚠️ Dataset file ('Dataset .csv') not found in project directory.")
    uploaded_file = st.sidebar.file_uploader("Upload Restaurant Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()
        df = df_raw.dropna(subset=['Latitude', 'Longitude']).copy()
    else:
        st.stop()

# ------------------------------------------------------------------------------
# 3. SIDEBAR FILTERS & INTERACTIVE CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Map & Analysis Filters")

# Filter by City
cities_list = ['All Cities'] + sorted(df['City'].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("Select City Filter", cities_list, index=0)

# Filter by Price Range
price_options = sorted(df['Price range'].dropna().unique().tolist())
selected_prices = st.sidebar.multiselect("Price Range Tier (1-4)", price_options, default=price_options)

# Filter by Minimum Rating
min_rating = st.sidebar.slider("Minimum Aggregate Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.5)

# Maximum Markers on Folium Map
max_map_markers = st.sidebar.slider("Maximum Map Markers (for performance)", min_value=100, max_value=5000, value=1000, step=100)

# Apply Filters to DataFrame
filtered_df = df.copy()
if selected_city != 'All Cities':
    filtered_df = filtered_df[filtered_df['City'] == selected_city]
if selected_prices:
    filtered_df = filtered_df[filtered_df['Price range'].isin(selected_prices)]
filtered_df = filtered_df[filtered_df['Aggregate rating'] >= min_rating]

# ------------------------------------------------------------------------------
# 4. DASHBOARD HEADER & KPI METRICS
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🗺️ Location-Based Restaurant Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatial mapping, geographic density metrics, price-rating correlation, and hidden gem discovery</div>', unsafe_allow_html=True)

# Overview KPIs
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.metric("Total Restaurants Mapped", f"{len(filtered_df):,}")
with col_kpi2:
    st.metric("Unique Cities Included", f"{filtered_df['City'].nunique()}")
with col_kpi3:
    st.metric("Avg Aggregate Rating", f"{filtered_df['Aggregate rating'].mean():.2f} ⭐")
with col_kpi4:
    st.metric("Avg Price Range", f"Tier {filtered_df['Price range'].mean():.2f}")

# ------------------------------------------------------------------------------
# 5. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Interactive Folium Map", 
    "🏙️ Market Concentration", 
    "📊 Price vs. Rating Analysis", 
    "💎 Potential Hidden Gems", 
    "📁 Dataset Explorer"
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE FOLIUM MAP
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("📍 Interactive Geospatial Restaurant Map")
    st.write(f"Displaying up to **{min(len(filtered_df), max_map_markers)}** clustered markers based on active filters:")

    if len(filtered_df) == 0:
        st.warning("No restaurants match the active filter criteria.")
    else:
        # Calculate central geographic coordinates
        map_center = [filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()]
        
        # Create Folium Map instance
        restaurant_map = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap")
        
        # Add MarkerCluster plugin
        marker_cluster = MarkerCluster().add_to(restaurant_map)

        # Subset data for map performance
        map_data = filtered_df.head(max_map_markers)

        for index, row in map_data.iterrows():
            popup_html = f"""
                <div style="font-family: sans-serif; min-width: 180px;">
                    <h4 style="margin: 0 0 6px 0; color: #1d3557;">{row['Restaurant Name']}</h4>
                    <b>City:</b> {row['City']}<br>
                    <b>Rating:</b> ⭐ {row['Aggregate rating']} / 5.0<br>
                    <b>Cuisines:</b> {row['Cuisines']}<br>
                    <b>Cost for 2:</b> {row.get('Currency', '$')} {row.get('Average Cost for two', 'N/A')}<br>
                    <b>Price Range:</b> Tier {row.get('Price range', 'N/A')}
                </div>
            """
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row['Restaurant Name']
            ).add_to(marker_cluster)

        # Render Folium Map inside Streamlit using HTML components
        components.html(restaurant_map._repr_html_(), height=550)

# ------------------------------------------------------------------------------
# TAB 2: MARKET CONCENTRATION BY CITY & LOCALITY
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🏙️ Restaurant Concentration by Location")
    
    col_conc1, col_conc2 = st.columns(2)

    with col_conc1:
        st.write("#### Top Cities by Restaurant Concentration")
        city_concentration = filtered_df['City'].value_counts().head(15).reset_index()
        city_concentration.columns = ['City', 'Restaurant Count']

        fig_city, ax_city = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=city_concentration, x='Restaurant Count', y='City', palette='viridis', ax=ax_city)
        ax_city.set_title('Top 15 Cities by Restaurant Count')
        ax_city.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_city)

    with col_conc2:
        st.write("#### Top 10 Localities by Concentration")
        locality_concentration = filtered_df['Locality'].value_counts().head(10).reset_index()
        locality_concentration.columns = ['Locality', 'Restaurant Count']

        fig_loc, ax_loc = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=locality_concentration, x='Restaurant Count', y='Locality', palette='magma', ax=ax_loc)
        ax_loc.set_title('Top 10 Localities by Restaurant Count')
        ax_loc.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_loc)

# ------------------------------------------------------------------------------
# TAB 3: PRICE VS. RATING STATISTICAL ANALYSIS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Price Range vs. Rating Insights")

    # Aggregate Statistics by City
    city_stats = filtered_df.groupby('City').agg({
        'Aggregate rating': 'mean',
        'Price range': 'mean',
        'Restaurant ID': 'count',
        'Cuisines': 'nunique'
    }).reset_index()

    city_stats.columns = ['City', 'Average Rating', 'Average Price Range', 'Restaurant Count', 'Unique Cuisines Count']
    city_stats = city_stats.sort_values(by='Average Rating', ascending=False)

    st.write("#### Location-Based Statistics (By City)")
    st.dataframe(city_stats, use_container_width=True)

    st.markdown("---")
    st.write("#### Average Rating vs. Price Range Scatter Plot")
    
    fig_scatter, ax_sc = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=city_stats, 
        x='Average Price Range', 
        y='Average Rating', 
        size='Restaurant Count', 
        hue='City',
        sizes=(100, 900),
        legend=False,
        ax=ax_sc
    )
    ax_sc.set_title('Average Rating vs. Price Range by City', fontsize=12, fontweight='bold')
    ax_sc.set_xlabel('Average Price Range Tier')
    ax_sc.set_ylabel('Average Aggregate Rating')
    ax_sc.grid(True, linestyle=':', alpha=0.5)
    st.pyplot(fig_scatter)

# ------------------------------------------------------------------------------
# TAB 4: HIDDEN GEMS & ONLINE DELIVERY PATTERNS
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("💎 Potential Hidden Gem Localities & Delivery Patterns")

    col_gem1, col_gem2 = st.columns(2)

    with col_gem1:
        st.write("#### Potential Hidden Gem Localities")
        st.write("Localities with **Average Rating > 4.0** and **Average Votes < 100**:")
        
        hidden_gems = filtered_df.groupby('Locality').agg({
            'Aggregate rating': 'mean', 
            'Votes': 'mean',
            'City': 'first',
            'Restaurant ID': 'count'
        }).reset_index()
        
        hidden_gems = hidden_gems[
            (hidden_gems['Aggregate rating'] > 4.0) & (hidden_gems['Votes'] < 100)
        ].sort_values(by='Aggregate rating', ascending=False)

        if len(hidden_gems) == 0:
            st.info("No localities fit the exact hidden gem criteria under active filters.")
        else:
            st.dataframe(hidden_gems.rename(columns={'Restaurant ID': 'Count'}), use_container_width=True)

    with col_gem2:
        st.write("#### Online Delivery Penetration by Locality")
        if 'Has Online delivery' in filtered_df.columns:
            delivery_pattern = filtered_df.groupby('Locality')['Has Online delivery'].value_counts(normalize=True).unstack().fillna(0)
            st.dataframe(delivery_pattern.head(15), use_container_width=True)
        else:
            st.write("Online delivery data not present in dataset.")

# ------------------------------------------------------------------------------
# TAB 5: DATASET EXPLORER
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("📁 Mapped Dataset Overview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

# ------------------------------------------------------------------------------
# 6. DIRECT SCRIPT EXECUTION SUPPORT (CLI BACKWARD COMPATIBILITY)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Location-Based Analysis Summary ---")
    print(f"Total Valid Coordinates Mapped: {len(df)} restaurants.")
    print("\nTop 5 Cities by Concentration:")
    print(df['City'].value_counts().head(5).to_string())
