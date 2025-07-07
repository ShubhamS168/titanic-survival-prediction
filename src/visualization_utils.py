"""
Visualization utilities for the deployment app
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_prediction_gauge(survival_probability):
    """
    Create a gauge chart for survival probability
    
    Args:
        survival_probability: Float between 0 and 1
        
    Returns:
        Plotly figure object
    """
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = survival_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Survival Probability (%)"},
        delta = {'reference': 50},
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
    return fig

def create_feature_impact_chart(feature_importance_data):
    """
    Create a feature impact chart
    
    Args:
        feature_importance_data: Dict or DataFrame with feature names and importance
        
    Returns:
        Plotly figure object
    """
    
    if isinstance(feature_importance_data, dict):
        df = pd.DataFrame(list(feature_importance_data.items()), 
                         columns=['feature', 'importance'])
    else:
        df = feature_importance_data
    
    # Sort by importance and take top 10
    df = df.sort_values('importance', ascending=False).head(10)
    
    fig = px.bar(
        df,
        x='importance',
        y='feature', 
        orientation='h',
        title="Feature Impact on Prediction",
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def create_confidence_indicator(confidence_score):
    """
    Create a confidence indicator visualization
    
    Args:
        confidence_score: Float between 0 and 1
        
    Returns:
        Plotly figure object
    """
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    confidence_ranges = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    color_idx = next(i for i, threshold in enumerate(confidence_ranges) 
                     if confidence_score <= threshold)
    
    fig = go.Figure(go.Indicator(
        mode = "number+gauge",
        value = confidence_score * 100,
        title = {'text': "Model Confidence (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': colors[color_idx]},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_model_performance_chart(results_data):
    """
    Create model performance comparison chart
    
    Args:
        results_data: DataFrame with model results
        
    Returns:
        Plotly figure object
    """
    
    fig = px.bar(
        results_data,
        x='model_name',
        y='test_accuracy',
        title="Model Performance Comparison",
        color='test_accuracy',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        xaxis_title="Model Type",
        yaxis_title="Test Accuracy",
        showlegend=False
    )
    
    return fig

def create_historical_comparison_chart(prediction_prob, historical_rates):
    """
    Create chart comparing prediction to historical rates
    
    Args:
        prediction_prob: Float, predicted survival probability
        historical_rates: Dict with historical survival rates
        
    Returns:
        Plotly figure object
    """
    
    data = {
        'Category': ['Your Prediction'] + list(historical_rates.keys()),
        'Survival Rate': [prediction_prob] + list(historical_rates.values()),
        'Type': ['Prediction'] + ['Historical'] * len(historical_rates)
    }
    
    fig = px.bar(
        pd.DataFrame(data),
        x='Category', 
        y='Survival Rate',
        color='Type',
        title="Prediction vs Historical Averages",
        color_discrete_map={'Prediction': '#1f77b4', 'Historical': '#ff7f0e'}
    )
    
    fig.update_layout(yaxis=dict(range=[0, 1]))
    return fig