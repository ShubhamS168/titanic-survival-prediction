"""
Model loading utilities for production deployment
"""

import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path
import os
# from src.model_loader import get_model_info

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model and metadata with comprehensive error handling"""
    try:
        # Check if model files exist
        model_path = 'models/best_model.pkl'
        features_path = 'models/feature_names.pkl' 
        metadata_path = 'models/model_metadata.json'
        
        if not all(Path(p).exists() for p in [model_path, features_path, metadata_path]):
            return None, None, None
        
        # Load model
        model = joblib.load(model_path)
        
        # Load feature names
        feature_names = joblib.load(features_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, feature_names, metadata
    
    except FileNotFoundError:
        st.error("""
        âŒ **Pre-trained model not found!**
        
        Please run the training script first:
        ```
        python scripts/train_and_save_models.py
        ```
        """)
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def get_model_info():
    """Get model information for display"""
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception:
        return None

def check_model_availability():
    """Check if all required model files are available"""
    required_files = [
        'models/best_model.pkl',
        'models/feature_names.pkl', 
        'models/model_metadata.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def get_model_version():
    """Get current model version"""
    try:
        metadata = get_model_info()
        return metadata.get('version', '1.0') if metadata else 'Unknown'
    except Exception:
        return 'Unknown'

@st.cache_data
def get_model_performance_summary():
    """Get cached model performance summary"""
    try:
        metadata = get_model_info()
        if metadata and 'all_results' in metadata:
            results_df = pd.DataFrame(metadata['all_results'])
            return {
                'best_accuracy': results_df['test_accuracy'].max(),
                'avg_accuracy': results_df['test_accuracy'].mean(),
                'model_count': len(results_df),
                'best_model': results_df.loc[results_df['test_accuracy'].idxmax(), 'model_name']
            }
    except Exception:
        pass
    return None