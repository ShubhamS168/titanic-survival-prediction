"""
Streamlit Model Deployment App - Titanic Survival Prediction
Production-ready deployment focused on pre-trained models
"""

import streamlit as st
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for deployment app
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .survived {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .not-survived {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Survival Prediction System</h1>
    <p>AI-Powered Passenger Survival Prediction | Deployed Model Ready</p>
</div>
""", unsafe_allow_html=True)

# Model status check
try:
    import sys
    import os
    sys.path.append('src')
    from model_loader import check_model_availability
    
    models_available, missing_files = check_model_availability()
    
    if models_available:
        st.markdown("""
        <div class="status-banner">
            ✅ <strong>Production Ready:</strong> Pre-trained models loaded and ready for deployment
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"""
        ❌ **Model files missing:** {', '.join(missing_files)}
        
        Please run the training script first:
        ```
        python scripts/train_high_performance.py
        ```
        """)
except Exception:
    st.warning("⚠️ Model status check unavailable. Ensure all source files are present.")

# Pages definition
PAGES = [
    st.Page(Path("pages/1_Home.py"), title=" Home", icon="🏠"),
    st.Page(Path("pages/2_Prediction.py"), title=" Make Prediction", icon="🎯"),
    st.Page(Path("pages/3_Model_Insights.py"), title=" Model Insights", icon="📊"),
    st.Page(Path("pages/4_About.py"), title=" About", icon="ℹ️"),
]

# Navigation
current_page = st.navigation(PAGES, position="sidebar", expanded=True)

# Add sidebar info
with st.sidebar:
    # st.markdown("---")
    st.markdown("### 🚀 Deployment Info")
    st.info("""
    **Production Model Deployment**
    
    - Instant predictions
    - Pre-trained models
    - Real-time analysis
    - Model explanations
    """)
    st.markdown("### 📊 Quick Stats")
    try:
        from src.model_loader import get_model_info
        metadata = get_model_info()
        if metadata:
            st.metric("Model", metadata['model_name'])
            st.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
            st.metric("Training Date", metadata['training_date'][:10])
    except:
        st.info("Model info will appear when available")

# Run current page
current_page.run()
