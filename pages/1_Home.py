"""
Home page for the Deployment-Focused Titanic Survival Prediction App
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

# ──────────────────────────────────────────────────────────────
#  Internal imports
# ──────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model_loader import get_model_info, check_model_availability
    from data_loader import load_titanic_data                 #  NEW
except ImportError:
    def get_model_info():
        return None
    def check_model_availability():
        return False, []
    def load_titanic_data():                                 #  Fallback
        return None

# ──────────────────────────────────────────────────────────────
#  Page header
# ──────────────────────────────────────────────────────────────
st.title("🚢 Welcome to Titanic Survival Predictor")
st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  Hero section
# ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ## Production ML Model Deployment

        Experience a **production-ready machine-learning system** that predicts passenger
        survival on the RMS Titanic. This deployment showcases real-world ML engineering
        practices with instant predictions, comprehensive model explanations, and
        interactive visualisations.

        ### 🎯 Key Features
        - 🏠 **Home** – Overview & quick start  
        - 🎯 **Make Prediction** – Real-time survival predictions  
        - 📊 **Model Insights** – Deep-dive performance & explanations  
        - ℹ️ **About** – Project & dataset details

        ### 🚀 Benefits
        - **Instant predictions** (cached model)  
        - **85 %+ accuracy** with robust validation  
        - **Confidence & explanation** for every prediction  
        - **Historical context** built-in
        """
    )

with col2:
    st.markdown("### 📈 System Status")
    models_available, missing_files = check_model_availability()

    if models_available:
        st.success("✅ **System Online**")
        meta = get_model_info()
        if meta:
            st.metric("Model",   meta["model_name"])
            st.metric("Accuracy", f"{meta['test_accuracy']:.1%}")
            st.metric("Features", meta["feature_count"])
            st.metric("Updated",  meta["training_date"][:10])
        if st.button("🚀 Make Your First Prediction", use_container_width=True):
            st.switch_page("pages/2_Prediction.py")
    else:
        st.error("❌ **Models Not Available**")
        st.info("Run `python scripts/train_high_performance.py` to initialise.")
        st.stop()

# ──────────────────────────────────────────────────────────────
#  NEW – Sample dataset preview (df.head())
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🗂️ Sample of Training Data")

data_preview = load_titanic_data()
if data_preview is not None:
    st.dataframe(data_preview.head(), use_container_width=True)
else:
    st.info(
        "Dataset preview unavailable – please ensure **data/train.csv** exists or "
        "run the training script to auto-download the Titanic dataset."
    )

# ──────────────────────────────────────────────────────────────
#  Model-performance overview
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🎯 Model Performance Overview")

meta = get_model_info()
if meta and "all_results" in meta:
    results_df = pd.DataFrame(meta["all_results"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            results_df,
            x="model_name",
            y="test_accuracy",
            title="Model Comparison During Training",
            color="test_accuracy",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Performance Metrics")
        display_df = results_df[["model_name", "test_accuracy", "cv_mean"]].round(3)
        display_df.columns = ["Model", "Test Acc", "CV Mean"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        best_row = results_df.iloc[results_df["test_accuracy"].idxmax()]
        st.success(f"🏆 **Best Model:** {best_row['model_name']} ({best_row['test_accuracy']:.1%})")
else:
    st.info("Model-performance data will appear once the system is initialised.")

# ──────────────────────────────────────────────────────────────
#  Historical insights, tech stack, call-to-action … (unchanged)
# ──────────────────────────────────────────────────────────────
# … keep the rest of your original sections here …

    
    # Historical performance data
    historical_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM'],
        'Expected Accuracy': ['82-84%', '83-85%', '81-83%', '62-70%'],
        'Strengths': [
            'Interpretable, fast',
            'Handles missing data well',
            'Robust performance',
            'Good with small datasets'
        ]
    }
    
    st.dataframe(pd.DataFrame(historical_data), use_container_width=True, hide_index=True)

# Key insights section
st.markdown("---")
st.markdown("## 🔍 Historical Survival Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 👥 Demographics
    - **Women first**: 74% of women survived vs 19% of men
    - **Children's advantage**: 52% survival rate for under-16s
    - **Age matters**: Younger passengers had better odds
    """)

with col2:
    st.markdown("""
    ### 💰 Economic Factors
    - **Class distinction**: 1st class 63%, 3rd class 24%
    - **Fare correlation**: Higher fares = better survival
    - **Cabin location**: Upper decks closer to lifeboats
    """)

with col3:
    st.markdown("""
    ### 🚢 Ship Factors
    - **Limited lifeboats**: Only enough for 1/3 of passengers
    - **Evacuation protocol**: "Women and children first"
    - **Disaster timeline**: ~2.5 hours from impact to sinking
    """)

# Technology showcase
st.markdown("---")
st.markdown("## 🛠️ Technology Stack")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.markdown("""
    ### 🤖 Machine Learning
    - **Scikit-learn**: Production ML pipeline
    - **XGBoost**: Advanced gradient boosting
    - **Feature Engineering**: 60+ engineered features
    - **Model Validation**: Stratified cross-validation
    """)

with tech_col2:
    st.markdown("""
    ### 🌐 Deployment
    - **Streamlit**: Interactive web interface
    - **Plotly**: Dynamic visualizations
    - **Joblib**: Efficient model serialization
    - **Production Ready**: Cached model loading
    """)

# Call to action
st.markdown("---")
st.markdown("## 🚀 Ready to Explore?")

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button("🎯 Make Prediction", use_container_width=True):
        st.switch_page("pages/2_Prediction.py")

with button_col2:
    if st.button("📊 View Model Insights", use_container_width=True):
        st.switch_page("pages/3_Model_Insights.py")

with button_col3:
    if st.button("ℹ️ Learn More", use_container_width=True):
        st.switch_page("pages/4_About.py")

# System requirements note
st.markdown("---")
st.info("""
**💡 Pro Tip:** This system demonstrates production ML deployment best practices including 
model versioning, caching, error handling, and real-time inference. Perfect for learning 
how to deploy machine learning models in real-world applications!
""")
