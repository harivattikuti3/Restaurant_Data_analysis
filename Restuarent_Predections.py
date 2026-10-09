# ==============================================================================
# RESTAURANT RATING PREDICTION SYSTEM & STREAMLIT WEB APP
# ==============================================================================
# Description:
#   This application trains a Machine Learning Pipeline (Linear Regression with 
#   ColumnTransformers for preprocessing numerical and categorical features) 
#   to predict restaurant aggregate ratings based on dataset features such as 
#   City, Cuisines, Cost, Votes, Table Booking, Online Delivery, etc.
#   It also provides an interactive Streamlit UI for user-driven predictions 
#   and model analytics.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Scikit-Learn Preprocessing & Pipeline Modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Scikit-Learn Model & Evaluation Metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced aesthetics and modern card design
st.markdown("""
    <style>
    /* Main container title styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Metric card design */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #FF6B6B;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2b2d42;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    /* Prediction Banner */
    .prediction-banner {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .prediction-score {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. DATA LOADING & CACHING FUNCTION
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """
    Loads restaurant dataset from local directory.
    Checks multiple potential file paths for robust execution.
    """
    candidate_paths = [
        os.path.join('dataset', 'Dataset .csv'),
        'Dataset .csv',
        os.path.join('dataset', 'Dataset.csv'),
        'Dataset.csv'
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.error(f"Error reading dataset at {path}: {e}")
                return None
    return None

# ------------------------------------------------------------------------------
# 3. MODEL BUILDING & TRAINING PIPELINE (CACHED)
# ------------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    """
    Preprocesses dataset, constructs Scikit-Learn Pipeline, fits model,
    and evaluates metrics.
    
    Parameters:
        df (pd.DataFrame): Input raw dataset
        
    Returns:
        tuple: (fitted_model, X_train, X_test, y_train, y_test, y_pred, mse, r2, coef_df, feature_cols)
    """
    # Define feature lists
    numerical_features = ['Votes', 'Price range', 'Average Cost for two']
    categorical_features = ['City', 'Cuisines', 'Has Table booking', 'Has Online delivery', 'Rating color', 'Rating text']
    all_features = categorical_features + numerical_features
    target_col = 'Aggregate rating'

    # Filter dataframe to required columns
    X = df[all_features]
    y = df[target_col]

    # Perform Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline for Numerical Features: Mean Imputation -> Standard Scaling
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for Categorical Features: Most Frequent Imputation -> One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine processing pipelines into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    # Complete Machine Learning Pipeline with Linear Regression Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train model on training data
    model.fit(X_train, y_train)

    # Generate predictions on test data
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Extract model coefficients and feature names
    regressor = model.named_steps['regressor'] 
    preprocessor_step = model.named_steps['preprocessor']
    feature_names = preprocessor_step.get_feature_names_out()
    coefficients = regressor.coef_

    # Create sorted DataFrame of feature importances
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    return model, X_train, X_test, y_train, y_test, y_pred, mse, r2, coef_df, all_features

# Load data and build model
df = load_dataset()

# Handle case when dataset is not found locally
if df is None:
    st.error("⚠️ Dataset file ('Dataset .csv') not found. Please ensure it is placed in the project directory.")
    uploaded_file = st.sidebar.file_uploader("Upload Restaurant Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Train model using cached function
model, X_train, X_test, y_train, y_test, y_pred, mse, r2, coef_df, feature_cols = train_model(df)

# ------------------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE & SIDEBAR INPUTS
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🍽️ Restaurant Rating Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict restaurant ratings using Machine Learning (Linear Regression with Scikit-Learn Pipeline)</div>', unsafe_allow_html=True)

# Sidebar Configuration for Prediction Inputs
st.sidebar.header("⚙️ Restaurant Feature Inputs")
st.sidebar.markdown("Provide restaurant parameters below to predict its **Aggregate Rating**.")

# Extract unique values for dynamic selectboxes
unique_cities = sorted(df['City'].dropna().unique().tolist())
unique_cuisines = sorted(df['Cuisines'].dropna().unique().tolist())
unique_colors = df['Rating color'].dropna().unique().tolist()
unique_texts = df['Rating text'].dropna().unique().tolist()

# Define sidebar form inputs with defaults
default_city = 'Makati City' if 'Makati City' in unique_cities else unique_cities[0]
selected_city = st.sidebar.selectbox("City", unique_cities, index=unique_cities.index(default_city))

# Cuisines selector/input
selected_cuisines = st.sidebar.selectbox("Cuisines", unique_cuisines, index=0)

# Binary options
table_booking = st.sidebar.radio("Has Table Booking?", ['Yes', 'No'], index=0)
online_delivery = st.sidebar.radio("Has Online Delivery?", ['No', 'Yes'], index=0)

# Numerical inputs
price_range = st.sidebar.select_slider("Price Range (1: Cheap, 4: Luxury)", options=[1, 2, 3, 4], value=3)
votes = st.sidebar.number_input("Total Votes", min_value=0, max_value=50000, value=314, step=10)
avg_cost = st.sidebar.number_input("Average Cost for Two", min_value=0, max_value=500000, value=600, step=50)

# Rating descriptors
rating_color = st.sidebar.selectbox("Rating Color Descriptor", unique_colors, index=0)
rating_text = st.sidebar.selectbox("Rating Text Descriptor", unique_texts, index=0)

# Prepare DataFrame for user single-sample prediction
input_data = pd.DataFrame({
    'City': [selected_city],
    'Cuisines': [selected_cuisines],
    'Has Table booking': [table_booking],
    'Has Online delivery': [online_delivery],
    'Price range': [price_range],
    'Votes': [votes],
    'Average Cost for two': [avg_cost],
    'Rating color': [rating_color],
    'Rating text': [rating_text]
})

# Predict button in sidebar
predict_clicked = st.sidebar.button("✨ Predict Restaurant Rating", use_container_width=True)

# ------------------------------------------------------------------------------
# 5. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Rating Predictor", 
    "📊 Model Performance", 
    "📈 Feature Importance", 
    "📁 Dataset Explorer"
])

# ------------------------------------------------------------------------------
# TAB 1: RATING PREDICTOR
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🎯 Single Restaurant Prediction")
    st.write("Current Input Configuration:")
    st.dataframe(input_data, use_container_width=True)
    
    # Calculate single prediction
    single_pred = model.predict(input_data)[0]
    # Bound rating between 0.0 and 5.0 for realistic output
    bounded_pred = round(max(0.0, min(5.0, single_pred)), 2)

    # Display prediction result banner
    st.markdown(f"""
        <div class="prediction-banner">
            <h3 style="margin:0; font-size:1.4rem;">Predicted Aggregate Rating</h3>
            <div class="prediction-score">⭐ {bounded_pred} / 5.0</div>
            <p style="margin:0; opacity:0.9;">Raw Model Output: {single_pred:.4f}</p>
        </div>
    """, unsafe_allow_html=True)

    # Explanation card
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Predicted Score", f"{bounded_pred} / 5.0")
    with col_b:
        st.metric("Price Range Tier", f"Tier {price_range}")
    with col_c:
        st.metric("Cost for Two", f"{avg_cost}")

# ------------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & METRICS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Regression Metrics & Evaluation")
    
    # Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")
    with col2:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col3:
        st.metric(label="Root Mean Sq Error (RMSE)", value=f"{np.sqrt(mse):.4f}")
    with col4:
        st.metric(label="Total Dataset Size", value=f"{len(df):,} records")

    st.markdown("---")
    st.write("### 📉 Actual vs. Predicted Ratings Scatter Plot")
    
    # Plot Actual vs Predicted scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, color='#2ec4b6', alpha=0.5, edgecolors='none', label='Predictions')
    # Ideal prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='#e71d36', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')
    
    ax.set_xlabel('Actual Ratings', fontsize=11)
    ax.set_ylabel('Predicted Ratings', fontsize=11)
    ax.set_title('Actual vs. Predicted Aggregate Ratings', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig)

# ------------------------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE & COEFFICIENTS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📈 Linear Regression Coefficients")
    st.write("Top positive and negative coefficient values extracted from the fitted pipeline model:")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.write("#### Feature Coefficients Table")
        st.dataframe(coef_df.reset_index(drop=True), height=400, use_container_width=True)

    with col_right:
        st.write("#### Top 15 Most Impactful Features")
        top_coefs = pd.concat([coef_df.head(8), coef_df.tail(7)]).drop_duplicates()
        fig_bar, ax_bar = plt.subplots(figsize=(7, 6))
        colors = ['#2ec4b6' if c > 0 else '#e71d36' for c in top_coefs['Coefficient']]
        ax_bar.barh(top_coefs['Feature'], top_coefs['Coefficient'], color=colors)
        ax_bar.set_xlabel('Coefficient Impact')
        ax_bar.set_title('Feature Importance (Positive vs Negative)')
        ax_bar.invert_yaxis()
        ax_bar.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_bar)

# ------------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📁 Dataset Preview & Overview")
    st.dataframe(df.head(50), use_container_width=True)
    
    st.markdown("### 📊 Dataset Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ------------------------------------------------------------------------------
# 6. DIRECT SCRIPT EXECUTION SUPPORT (CLI BACKWARD COMPATIBILITY)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Log model metrics to terminal for CLI execution
    print("\n--- Model Training & Evaluation Summary ---")
    print(f"Model R-squared (R²): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    print("\nTop 5 Positive Feature Coefficients:")
    print(coef_df.head(5).to_string(index=False))
