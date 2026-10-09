# ==============================================================================
# RESTAURANT CUISINE CLASSIFICATION SYSTEM & STREAMLIT WEB APP
# ==============================================================================
# Description:
#   This application trains Machine Learning Classification models (Logistic Regression 
#   and Random Forest Classifier) to predict the primary cuisine of a restaurant based 
#   on location features (Country Code, City, Locality, Locality Verbose).
#   It includes a complete Streamlit dashboard for model performance evaluation, 
#   metric comparisons, and user-driven single-sample cuisine predictions.
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

# Scikit-Learn Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Scikit-Learn Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

# ------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Cuisine Classifier App",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling and modern cards
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4EA8DE, #5390D9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .cuisine-banner {
        background: linear-gradient(135deg, #4EA8DE, #7400B8);
        color: white;
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .cuisine-name {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 6px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #5390D9;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING (CACHED)
# ------------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    """
    Loads raw restaurant dataset, strips whitespace from column headers,
    extracts primary cuisine, and removes rows with missing cuisine labels.
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
        return None, None, None, None

    # Remove extra spaces from column names
    df_raw.columns = df_raw.columns.str.strip()

    # Extract primary cuisine (first item before comma)
    df = df_raw.copy()
    df['Cuisines'] = df['Cuisines'].astype(str).str.split(',').str[0].str.strip()
    df = df.dropna(subset=['Cuisines'])
    df = df[df['Cuisines'] != 'nan']

    # Define Feature matrix X and Target vector y
    feature_cols = ['Country Code', 'City', 'Locality', 'Locality Verbose']
    X = df[feature_cols].copy()
    y = df['Cuisines'].copy()

    return df, X, y, feature_cols

# Load cleaned dataset
df, X, y, feature_cols = load_and_clean_data()

# Handle case when dataset file is missing
if df is None:
    st.error("⚠️ Dataset file ('Dataset .csv') not found in project directory.")
    uploaded_file = st.sidebar.file_uploader("Upload Restaurant Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()
        df = df_raw.copy()
        df['Cuisines'] = df['Cuisines'].astype(str).str.split(',').str[0].str.strip()
        df = df.dropna(subset=['Cuisines'])
        X = df[['Country Code', 'City', 'Locality', 'Locality Verbose']]
        y = df['Cuisines']
    else:
        st.stop()

# ------------------------------------------------------------------------------
# 3. MODEL BUILDING & EVALUATION PIPELINES (CACHED)
# ------------------------------------------------------------------------------
@st.cache_resource
def train_and_evaluate_models(X, y):
    """
    Constructs preprocessing pipelines, trains Logistic Regression and 
    Random Forest Classifiers, and evaluates performance metrics.
    
    Returns:
        tuple: (lr_model, rf_model, metrics_df, X_train, X_test, y_train, y_test, lr_report, rf_report)
    """
    # Feature columns split
    num_data = ['Country Code']
    cat_data = ['City', 'Locality', 'Locality Verbose']

    # Numerical Pipeline: Mean Imputation -> Standard Scaling
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline: Most Frequent Imputation -> One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combined Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_data),
        ('cat', cat_pipeline, cat_data)
    ])

    # Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Logistic Regression Model Pipeline
    lr_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # Evaluate Logistic Regression
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec = precision_score(y_test, lr_pred, average='weighted', zero_division=0)
    lr_rec = recall_score(y_test, lr_pred, average='weighted', zero_division=0)
    lr_f1 = f1_score(y_test, lr_pred, average='weighted', zero_division=0)
    lr_report = classification_report(y_test, lr_pred, zero_division=0)

    # 2. Random Forest Classifier Pipeline
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Evaluate Random Forest
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)
    rf_report = classification_report(y_test, rf_pred, zero_division=0)

    # Metrics comparison summary DataFrame
    metrics_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest Classifier'],
        'Accuracy': [lr_acc, rf_acc],
        'Precision': [lr_prec, rf_prec],
        'Recall': [lr_rec, rf_rec],
        'F1-Score': [lr_f1, rf_f1]
    })

    return lr_model, rf_model, metrics_df, X_train, X_test, y_train, y_test, lr_report, rf_report

# Train models using cached execution
lr_model, rf_model, metrics_df, X_train, X_test, y_train, y_test, lr_report, rf_report = train_and_evaluate_models(X, y)

# ------------------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE & SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🍜 Restaurant Cuisine Classifier App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classify primary restaurant cuisines based on location parameters using Machine Learning (Logistic Regression & Random Forest)</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Classification Inputs")

# Model Selection
selected_model_name = st.sidebar.radio(
    "Choose Classification Model:",
    ["Random Forest Classifier (Recommended)", "Logistic Regression"],
    index=0
)
active_model = rf_model if "Random Forest" in selected_model_name else lr_model

# Extract unique location features for dropdowns
unique_country_codes = sorted(df['Country Code'].dropna().unique().tolist())
unique_cities = sorted(df['City'].dropna().unique().tolist())
unique_localities = sorted(df['Locality'].dropna().unique().tolist())
unique_locality_verboses = sorted(df['Locality Verbose'].dropna().unique().tolist())

# Selectbox controls
selected_country = st.sidebar.selectbox("Country Code", unique_country_codes, index=0)
selected_city = st.sidebar.selectbox("City", unique_cities, index=0)

# Filter localities matching selected city if possible
city_filtered_localities = sorted(df[df['City'] == selected_city]['Locality'].dropna().unique().tolist())
locality_options = city_filtered_localities if len(city_filtered_localities) > 0 else unique_localities
selected_locality = st.sidebar.selectbox("Locality", locality_options, index=0)

# Filter locality verbose matching selected locality if possible
locality_filtered_verboses = sorted(df[df['Locality'] == selected_locality]['Locality Verbose'].dropna().unique().tolist())
verbose_options = locality_filtered_verboses if len(locality_filtered_verboses) > 0 else unique_locality_verboses
selected_verbose = st.sidebar.selectbox("Locality Verbose", verbose_options, index=0)

# Single sample input DataFrame
input_df = pd.DataFrame([{
    'Country Code': selected_country,
    'City': selected_city,
    'Locality': selected_locality,
    'Locality Verbose': selected_verbose
}])

# Predict Button
predict_btn = st.sidebar.button("✨ Classify Cuisine", use_container_width=True)

# ------------------------------------------------------------------------------
# 5. DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Cuisine Predictor", 
    "📊 Model Comparison", 
    "📈 Cuisine Distribution", 
    "📁 Dataset Explorer"
])

# ------------------------------------------------------------------------------
# TAB 1: CUISINE PREDICTOR
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🎯 Single Restaurant Cuisine Prediction")
    st.write("Selected Location Input Parameters:")
    st.dataframe(input_df, use_container_width=True)

    # Perform Prediction
    predicted_cuisine = active_model.predict(input_df)[0]

    st.markdown(f"""
        <div class="cuisine-banner">
            <h3 style="margin:0; font-size:1.4rem;">Predicted Primary Cuisine</h3>
            <div class="cuisine-name">🍲 {predicted_cuisine}</div>
            <p style="margin:0; opacity:0.9;">Model Used: {selected_model_name}</p>
        </div>
    """, unsafe_allow_html=True)

    # Display Probability distribution if available
    try:
        classifier_step = active_model.named_steps['classifier']
        if hasattr(classifier_step, "predict_proba"):
            probs = active_model.predict_proba(input_df)[0]
            classes = classifier_step.classes_
            prob_df = pd.DataFrame({
                'Cuisine': classes,
                'Probability': probs
            }).sort_values(by='Probability', ascending=False).head(5)

            st.write("#### 🔝 Top 5 Likely Cuisines & Probability Scores")
            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
    except Exception:
        pass

# ------------------------------------------------------------------------------
# TAB 2: MODEL COMPARISON & METRICS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Classifier Performance Comparison")
    st.write("Comparative evaluation of Logistic Regression vs. Random Forest Classifier on test set data (20% split):")

    # Display Metrics Table
    st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Model Accuracy & F1-Score Comparison")
        fig_metrics, ax_m = plt.subplots(figsize=(6, 4))
        x_indices = np.arange(len(metrics_df))
        width = 0.35
        ax_m.bar(x_indices - width/2, metrics_df['Accuracy'], width, label='Accuracy', color='#4EA8DE')
        ax_m.bar(x_indices + width/2, metrics_df['F1-Score'], width, label='F1-Score', color='#7400B8')
        ax_m.set_xticks(x_indices)
        ax_m.set_xticklabels(metrics_df['Model'])
        ax_m.set_ylabel('Score')
        ax_m.set_ylim(0, 1.0)
        ax_m.legend()
        ax_m.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_metrics)

    with col2:
        st.write("#### Classification Metrics Breakdown")
        rf_row = metrics_df[metrics_df['Model'].str.contains('Random Forest')].iloc[0]
        st.metric("Random Forest Accuracy", f"{rf_row['Accuracy']:.4f}")
        st.metric("Random Forest Precision", f"{rf_row['Precision']:.4f}")
        st.metric("Random Forest Recall", f"{rf_row['Recall']:.4f}")
        st.metric("Random Forest F1-Score", f"{rf_row['F1-Score']:.4f}")

    st.markdown("---")
    st.write("#### 📜 Classification Reports")
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.write("**Logistic Regression Report:**")
        st.code(lr_report)
    with col_rep2:
        st.write("**Random Forest Report:**")
        st.code(rf_report)

# ------------------------------------------------------------------------------
# TAB 3: CUISINE DISTRIBUTION
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📈 Top Cuisines Distribution in Dataset")
    
    top_n = st.slider("Select Top N Cuisines to Display", min_value=5, max_value=30, value=15)
    cuisine_counts = df['Cuisines'].value_counts().head(top_n)

    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, palette='crest', ax=ax_c)
    ax_c.set_xlabel('Number of Restaurants')
    ax_c.set_ylabel('Cuisine')
    ax_c.set_title(f'Top {top_n} Primary Cuisines in Dataset')
    ax_c.grid(True, linestyle=':', alpha=0.5)
    st.pyplot(fig_c)

# ------------------------------------------------------------------------------
# TAB 4: DATASET EXPLORER
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📁 Dataset Preview & Overview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write(f"Total Processed Dataset Records: **{len(df):,} restaurants** across **{df['Cuisines'].nunique()} unique cuisines**.")

# ------------------------------------------------------------------------------
# 6. DIRECT SCRIPT EXECUTION SUPPORT (CLI BACKWARD COMPATIBILITY)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Cuisine Classification Model Summary ---")
    print(metrics_df.to_string(index=False))
