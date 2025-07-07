"""
Main prediction interface - Production deployment focused
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from model_loader import load_trained_model, get_model_info
    from data_processor import process_user_input
    from visualization_utils import create_prediction_gauge, create_feature_impact_chart
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all source files are available in the src/ directory.")
    st.stop()

st.title("🎯 Titanic Survival Prediction")
st.markdown("### *Real-time AI-powered survival analysis*")
st.markdown("---")

# Load model with error handling
try:
    model, feature_names, metadata = load_trained_model()
    
    if model is None:
        st.error("""
        ❌ **Model not available**
        
        Please run the training script first:
        ```
        python scripts/train_high_performance.py
        ```
        """)
        st.stop()
        
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Display model info header
st.markdown("## 🤖 Model Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Type", metadata['model_name'])
with col2:
    st.metric("Accuracy", f"{metadata['test_accuracy']:.1%}")
with col3:
    st.metric("Features", metadata['feature_count'])
with col4:
    st.metric("Training Date", metadata['training_date'][:10])

st.markdown("---")

# Prediction interface
st.markdown("## 👤 Enter Passenger Details")

# Example passengers for quick testing
st.markdown("### 🎲 Quick Examples")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("👩 First Class Woman"):
        st.session_state.update({
            'example_pclass': 1, 'example_sex': 'female', 'example_age': 35,
            'example_sibsp': 1, 'example_parch': 0, 'example_fare': 80.0,
            'example_embarked': 'S'
        })

with example_col2:
    if st.button("👨 Third Class Man"):
        st.session_state.update({
            'example_pclass': 3, 'example_sex': 'male', 'example_age': 25,
            'example_sibsp': 0, 'example_parch': 0, 'example_fare': 8.0,
            'example_embarked': 'S'
        })

with example_col3:
    if st.button("👶 Child with Family"):
        st.session_state.update({
            'example_pclass': 2, 'example_sex': 'female', 'example_age': 8,
            'example_sibsp': 1, 'example_parch': 2, 'example_fare': 25.0,
            'example_embarked': 'C'
        })

# Main input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Personal Information")
        
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            index=st.session_state.get('example_pclass', 3) - 1,
            format_func=lambda x: f"Class {x} ({'First' if x==1 else 'Second' if x==2 else 'Third'})",
            help="Ticket class - 1st class was most expensive with better amenities"
        )
        
        sex = st.selectbox(
            "Gender", 
            options=['male', 'female'],
            index=0 if st.session_state.get('example_sex', 'male') == 'male' else 1,
            help="Passenger gender - historically significant factor in survival"
        )
        
        age = st.slider(
            "Age", 
            min_value=0, 
            max_value=100, 
            value=st.session_state.get('example_age', 30),
            help="Passenger age in years - children had priority in evacuation"
        )
        
        embarked = st.selectbox(
            "Port of Embarkation",
            options=['S', 'C', 'Q'],
            index=['S', 'C', 'Q'].index(st.session_state.get('example_embarked', 'S')),
            format_func=lambda x: {
                'S': 'Southampton, England', 
                'C': 'Cherbourg, France', 
                'Q': 'Queenstown, Ireland'
            }[x],
            help="Where the passenger boarded the ship"
        )
    
    with col2:
        st.markdown("### 👨‍👩‍👧‍👦 Family & Ticket Details")
        
        sibsp = st.number_input(
            "Siblings/Spouses Aboard",
            min_value=0, 
            max_value=10, 
            value=st.session_state.get('example_sibsp', 0),
            help="Number of siblings or spouses traveling together"
        )
        
        parch = st.number_input(
            "Parents/Children Aboard",
            min_value=0, 
            max_value=10, 
            value=st.session_state.get('example_parch', 0),
            help="Number of parents or children traveling together"
        )
        
        fare = st.number_input(
            "Ticket Fare (£)",
            min_value=0.0, 
            max_value=1000.0, 
            value=float(st.session_state.get('example_fare', 32.0)),
            step=0.1,
            help="Ticket price in British pounds (1912 currency)"
        )
        
        # Optional advanced fields
        with st.expander("🔧 Advanced Options (Optional)"):
            cabin = st.text_input("Cabin Number", help="Leave blank if unknown")
            ticket = st.text_input("Ticket Number", value="UNKNOWN")
            name = st.text_input("Passenger Name", value="Smith, Mr. John")
    
    # Prediction button
    submitted = st.form_submit_button("🎯 Predict Survival", type="primary", use_container_width=True)

# Clear example state after form submission
if submitted:
    for key in list(st.session_state.keys()):
        if key.startswith('example_'):
            del st.session_state[key]

# Process prediction
if submitted:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'Cabin': [cabin if cabin else np.nan],
        'Ticket': [ticket],
        'Name': [name]
    })
    
    try:
        # Process input and make prediction
        processed_data = process_user_input(input_data)
        
        # Debug: Show feature shapes for troubleshooting
        if st.checkbox("Show Debug Info", value=False):
            st.write(f"Processed data shape: {processed_data.shape}")
            st.write(f"Processed features: {list(processed_data.columns)}")
            
            # Load expected features
            try:
                import joblib
                expected_features = joblib.load('models/feature_names.pkl')
                st.write(f"Expected features count: {len(expected_features)}")
                
                missing_in_processed = [f for f in expected_features if f not in processed_data.columns]
                extra_in_processed = [f for f in processed_data.columns if f not in expected_features]
                
                if missing_in_processed:
                    st.error(f"Missing features: {missing_in_processed[:10]}...")
                if extra_in_processed:
                    st.warning(f"Extra features: {extra_in_processed[:10]}...")
                    
            except Exception as e:
                st.error(f"Could not load feature names: {e}")
        
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Main prediction display
        if prediction == 1:
            st.markdown("""
            <div class="prediction-result survived">
                ✅ SURVIVED
            </div>
            """, unsafe_allow_html=True)
            outcome = "SURVIVED"
            survival_prob = probabilities[1]
        else:
            st.markdown("""
            <div class="prediction-result not-survived">
                ❌ DID NOT SURVIVE
            </div>
            """, unsafe_allow_html=True)
            outcome = "DID NOT SURVIVE"
            survival_prob = probabilities[1]
        
        # Detailed results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = survival_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Survival Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence metrics
            st.markdown("### 📊 Prediction Details")
            st.metric("Survival Probability", f"{survival_prob:.1%}")
            st.metric("Model Confidence", f"{max(probabilities):.1%}")
            st.metric("Family Size", sibsp + parch + 1)
            
            # Confidence interpretation
            confidence = max(probabilities)
            if confidence > 0.8:
                st.success("🔥 **High Confidence**")
                st.caption("The model is very confident in this prediction")
            elif confidence > 0.6:
                st.warning("⚡ **Medium Confidence**")
                st.caption("The model has moderate confidence")
            else:
                st.error("❓ **Low Confidence**")
                st.caption("The model is uncertain about this prediction")
        
        # Feature impact analysis
        st.markdown("### 🧠 What Influenced This Prediction?")

        def get_feature_importance(model, feature_names):
            """Extract feature importance from various model types"""
            
            try:
                # Case 1: Pipeline with classifier that has feature_importances_
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    classifier = model.named_steps['classifier']
                    
                    # Get feature names after preprocessing
                    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                        feature_names_processed = model.named_steps['preprocessor'].get_feature_names_out()
                    else:
                        feature_names_processed = feature_names
                    
                    # Check different types of classifiers
                    if hasattr(classifier, 'feature_importances_'):
                        # Tree-based models (RandomForest, XGBoost, etc.)
                        return feature_names_processed, classifier.feature_importances_
                        
                    elif hasattr(classifier, 'coef_'):
                        # Linear models (LogisticRegression)
                        importances = np.abs(classifier.coef_[0])
                        return feature_names_processed, importances
                        
                    elif hasattr(classifier, 'estimators_'):
                        # Ensemble models (VotingClassifier)
                        if hasattr(classifier.estimators_[0], 'feature_importances_'):
                            # Average feature importances from tree-based estimators
                            importances_list = []
                            for estimator in classifier.estimators_:
                                if hasattr(estimator, 'feature_importances_'):
                                    importances_list.append(estimator.feature_importances_)
                            
                            if importances_list:
                                avg_importances = np.mean(importances_list, axis=0)
                                return feature_names_processed, avg_importances
                
                # Case 2: Direct model with feature_importances_
                elif hasattr(model, 'feature_importances_'):
                    return feature_names, model.feature_importances_
                    
                # Case 3: Direct model with coefficients
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0])
                    return feature_names, importances
                    
            except Exception as e:
                print(f"Error extracting feature importance: {e}")
                return None, None
            
            return None, None

        # Try to get feature importance
        feature_names_processed, importances = get_feature_importance(model, feature_names)

        if importances is not None and len(importances) > 0:
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names_processed,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Display top 10 features
            top_features = importance_df.head(10)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create horizontal bar chart
                fig = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    labels={'importance': 'Feature Importance', 'feature': 'Features'}
                )
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Top Features")
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    st.write(f"{i}. **{row['feature']}**: {row['importance']:.3f}")

        else:
            # Fallback: Show general influence factors
            st.info("""
            **Feature importance not directly available, but here are the key factors that typically influence Titanic survival predictions:**
            
            🎯 **Primary Factors:**
            - **Gender**: Women had 74% survival rate vs 19% for men
            - **Passenger Class**: 1st class (63%), 2nd class (47%), 3rd class (24%)
            - **Age**: Children under 16 had 52% survival rate
            
            📊 **Secondary Factors:**
            - **Family Size**: Small families (2-4) had better survival odds
            - **Fare**: Higher fares correlated with better survival
            - **Cabin Location**: Upper decks had better evacuation access
            - **Port of Embarkation**: Reflected passenger demographics
            
            💡 **Your Prediction Analysis:**
            Based on your inputs, the model likely weighted these factors according to historical survival patterns.
            """)
            
            # Show prediction factors for this specific case
            st.markdown("#### 🔍 Factors for Your Prediction")
            
            factors = []
            if sex == 'female':
                factors.append("✅ **Female passenger** (+55% survival advantage)")
            else:
                factors.append("❌ **Male passenger** (-36% survival disadvantage)")
            
            if pclass == 1:
                factors.append("✅ **First class** (+25% survival advantage)")
            elif pclass == 2:
                factors.append("⚡ **Second class** (+9% survival advantage)")
            else:
                factors.append("❌ **Third class** (baseline survival rate)")
            
            if age < 16:
                factors.append("✅ **Child passenger** (+14% survival advantage)")
            elif age > 60:
                factors.append("❌ **Elderly passenger** (-5% survival disadvantage)")
            
            family_size = sibsp + parch + 1
            if 2 <= family_size <= 4:
                factors.append("✅ **Optimal family size** (+10% survival advantage)")
            elif family_size > 4:
                factors.append("❌ **Large family** (-15% survival disadvantage)")
            elif family_size == 1:
                factors.append("⚡ **Traveling alone** (neutral factor)")
            
            for factor in factors:
                st.markdown(factor)

        
        # Historical context
        st.markdown("### 📚 Historical Context")
        
        col1, col2 = st.columns(2)
        
        # Port mapping for cleaner code
        port_mapping = {
            'S': 'Southampton, England',
            'C': 'Cherbourg, France', 
            'Q': 'Queenstown, Ireland'
        }
        
        with col1:
            st.markdown(f"""
            **Your Passenger Profile:**
            - **Class:** {pclass} ({'Upper' if pclass==1 else 'Middle' if pclass==2 else 'Lower'} class)
            - **Demographics:** {age}-year-old {sex}
            - **Family:** {sibsp + parch} relatives aboard
            - **Ticket Fare:** £{fare:.2f}
            - **Embarked:** {port_mapping.get(embarked, 'Unknown Port')}
            """)
        
        with col2:
            # Get historical survival rates for comparison
            historical_rates = {
                'Overall': 0.38,
                'Women': 0.74 if sex == 'female' else 0.19,
                'Children (under 16)': 0.52 if age < 16 else None,
                f'Class {pclass}': {1: 0.63, 2: 0.47, 3: 0.24}[pclass]
            }
            
            st.markdown("**Historical Survival Rates:**")
            for category, rate in historical_rates.items():
                if rate is not None:
                    st.write(f"- **{category}:** {rate:.0%}")
        
        # Comparison with historical averages
        your_group_rate = historical_rates[f'Class {pclass}']
        if sex == 'female':
            your_group_rate = max(your_group_rate, historical_rates['Women'])
        
        st.markdown("### 📈 Prediction vs Historical Average")
        
        comparison_data = {
            'Category': ['Your Prediction', f'Historical {sex.title()} Class {pclass}'],
            'Survival Rate': [survival_prob, your_group_rate]
        }
        
        fig = px.bar(
            pd.DataFrame(comparison_data),
            x='Category',
            y='Survival Rate',
            title="Your Prediction vs Historical Group Average",
            color=['Prediction', 'Historical'],
            color_discrete_map={'Prediction': '#1f77b4', 'Historical': '#ff7f0e'}
        )
        fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.error("Please check your input values and try again.")
        
        # Enhanced error details
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            
            # Try to diagnose the issue
            try:
                import joblib
                expected_features = joblib.load('models/feature_names.pkl')
                processed_data = process_user_input(input_data)
                
                st.write("**Feature Comparison:**")
                st.write(f"Expected: {len(expected_features)} features")
                st.write(f"Got: {len(processed_data.columns)} features")
                
                missing = [f for f in expected_features if f not in processed_data.columns]
                extra = [f for f in processed_data.columns if f not in expected_features]
                
                if missing:
                    st.write(f"**Missing features:** {missing}")
                if extra:
                    st.write(f"**Extra features:** {extra}")
                    
            except Exception as debug_error:
                st.write(f"Debug failed: {debug_error}")

# Model limitations disclaimer
st.markdown("---")
st.warning("""
### ⚠️ Important Disclaimer

This prediction is based on a machine learning model trained on historical data from the 1912 Titanic disaster. 

**Key limitations:**
- **Historical context**: Based on 1912 maritime disaster conditions
- **Limited features**: Uses only passenger manifest data
- **Model accuracy**: ~80% accuracy means predictions can be incorrect
- **Educational purpose**: This tool is for learning and historical analysis

**Remember**: Real survival depended on many factors not captured in passenger data, including exact location during the disaster, individual actions, crew assistance, and chance.
""")

st.markdown("---")
st.markdown("*Try different passenger profiles to explore how various factors influenced survival on the Titanic.*")
