"""
Data processing utilities for deployment - FIXED VERSION
Ensures feature engineering matches training pipeline
"""

import pandas as pd
import numpy as np
import sys
import os

# Import the SAME feature engineering used in training
try:
    from feature_engineeringAfter import build_features
except ImportError:
    # Fallback to regular feature engineering if After version not available
    from feature_engineering import build_features

def process_user_input(input_data):
    """
    Process user input data for prediction - FIXED VERSION
    Ensures proper categorical encoding
    """
    
    # Convert to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    try:
        # Apply feature engineering first
        processed_df = build_features(df)
        
        # Load the feature names used during training
        import joblib
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Ensure we have all required features
        missing_features = [f for f in feature_names if f not in processed_df.columns]
        for feat in missing_features:
            # processed_df[feat] = 0  # Fill missing features with default values
            processed_df.loc[:, feat] = 0

        
        # Select only the features used during training
        final_df = processed_df[feature_names]
        
        # CRITICAL: Encode categorical variables
        categorical_cols = final_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in final_df.columns:
                # Simple label encoding for categorical variables
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                
                # Handle the encoding based on common patterns
                if col == 'Sex' or 'Sex' in col:
                    # For Sex: male=0, female=1
                    final_df[col] = final_df[col].map({'male': 0, 'female': 1}).fillna(0)
                elif col == 'Embarked' or 'Embarked' in col:
                    # For Embarked: S=0, C=1, Q=2
                    final_df[col] = final_df[col].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)
                else:
                    # For other categorical columns, use label encoding
                    unique_vals = final_df[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    final_df[col] = final_df[col].map(mapping).fillna(0)
        
        # Ensure all columns are numeric
        final_df = final_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return final_df
        
    except Exception as e:
        print(f"Error in process_user_input: {e}")
        
        # Fallback: Manual encoding
        df_encoded = df.copy()
        
        # Manual encoding for critical columns
        if 'Sex' in df_encoded.columns:
            df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
        
        if 'Embarked' in df_encoded.columns:
            df_encoded['Embarked'] = df_encoded['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Fill any remaining categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_encoded[col] = 0  # Replace with default value
        
        return df_encoded

def validate_input_data(data):
    """
    Validate user input data
    
    Args:
        data: Dictionary with user input
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    validations = [
        ('Pclass', lambda x: x in [1, 2, 3], "Passenger class must be 1, 2, or 3"),
        ('Sex', lambda x: x in ['male', 'female'], "Sex must be 'male' or 'female'"),
        ('Age', lambda x: 0 <= x <= 100, "Age must be between 0 and 100"),
        ('SibSp', lambda x: 0 <= x <= 10, "Siblings/spouses must be between 0 and 10"),
        ('Parch', lambda x: 0 <= x <= 10, "Parents/children must be between 0 and 10"),
        ('Fare', lambda x: x >= 0, "Fare must be non-negative"),
        ('Embarked', lambda x: x in ['S', 'C', 'Q'], "Embarked must be 'S', 'C', or 'Q'")
    ]
    
    for field, validator, error_msg in validations:
        if not validator(data[field]):
            return False, error_msg
    
    return True, "Input data is valid"

def create_example_inputs():
    """Create example input data for testing"""
    
    examples = {
        'first_class_woman': {
            'Pclass': 1,
            'Sex': 'female', 
            'Age': 35,
            'SibSp': 1,
            'Parch': 0,
            'Fare': 80.0,
            'Embarked': 'S',
            'Cabin': 'C85',
            'Ticket': 'PC 17599',
            'Name': 'Smith, Mrs. John (Mary)'
        },
        'third_class_man': {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 25, 
            'SibSp': 0,
            'Parch': 0,
            'Fare': 8.0,
            'Embarked': 'S',
            'Cabin': np.nan,
            'Ticket': '12345',
            'Name': 'Jones, Mr. William'
        },
        'child_with_family': {
            'Pclass': 2,
            'Sex': 'female',
            'Age': 8,
            'SibSp': 1,
            'Parch': 2, 
            'Fare': 25.0,
            'Embarked': 'C',
            'Cabin': np.nan,
            'Ticket': '54321', 
            'Name': 'Wilson, Miss. Emma'
        }
    }
    
    return examples



# """
# Data processing utilities for deployment - FIXED VERSION
# Ensures feature engineering matches training pipeline
# """

# import pandas as pd
# import numpy as np
# import sys
# import os

# # Import the SAME feature engineering used in training
# # try:
# #     from feature_engineering import build_features
# # except ImportError:
# #     # Fallback to regular feature engineering if After version not available
# #     from feature_engineering import build_features
    
# from feature_engineering import build_features

# def process_user_input(input_data):
#     """
#     Process user input data for prediction - FIXED VERSION
#     Now uses the same feature engineering as training
    
#     Args:
#         input_data: Dictionary or DataFrame with user input
        
#     Returns:
#         Processed DataFrame ready for model prediction
#     """
    
#     # Convert to DataFrame if needed
#     if isinstance(input_data, dict):
#         df = pd.DataFrame([input_data])
#     else:
#         df = input_data.copy()
    
#     # Apply the SAME feature engineering used in training
#     try:
#         processed_df = build_features(df)
        
#         # Ensure we only return features that existed during training
#         # Load the saved feature names to ensure consistency
#         try:
#             import joblib
#             feature_names = joblib.load('models/feature_names.pkl')
            
#             # Filter to only include features that were used in training
#             available_features = [col for col in feature_names if col in processed_df.columns]
#             missing_features = [col for col in feature_names if col not in processed_df.columns]
            
#             if missing_features:
#                 print(f"Warning: Missing features during prediction: {missing_features}")
#                 # Fill missing features with default values
#                 for feat in missing_features:
#                     processed_df[feat] = 0
            
#             # Return only the features used during training, in the same order
#             final_df = processed_df[feature_names]
#             return final_df
            
#         except Exception as e:
#             print(f"Error loading feature names: {e}")
#             return processed_df
            
#     except Exception as e:
#         print(f"Feature engineering failed: {e}")
#         # Return basic processing if advanced features fail
#         return df

# def validate_input_data(data):
#     """
#     Validate user input data
    
#     Args:
#         data: Dictionary with user input
        
#     Returns:
#         Tuple of (is_valid, error_message)
#     """
    
#     required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
#     # Check required fields
#     for field in required_fields:
#         if field not in data:
#             return False, f"Missing required field: {field}"
    
#     # Validate ranges
#     validations = [
#         ('Pclass', lambda x: x in [1, 2, 3], "Passenger class must be 1, 2, or 3"),
#         ('Sex', lambda x: x in ['male', 'female'], "Sex must be 'male' or 'female'"),
#         ('Age', lambda x: 0 <= x <= 100, "Age must be between 0 and 100"),
#         ('SibSp', lambda x: 0 <= x <= 10, "Siblings/spouses must be between 0 and 10"),
#         ('Parch', lambda x: 0 <= x <= 10, "Parents/children must be between 0 and 10"),
#         ('Fare', lambda x: x >= 0, "Fare must be non-negative"),
#         ('Embarked', lambda x: x in ['S', 'C', 'Q'], "Embarked must be 'S', 'C', or 'Q'")
#     ]
    
#     for field, validator, error_msg in validations:
#         if not validator(data[field]):
#             return False, error_msg
    
#     return True, "Input data is valid"

# def create_example_inputs():
#     """Create example input data for testing"""
    
#     examples = {
#         'first_class_woman': {
#             'Pclass': 1,
#             'Sex': 'female', 
#             'Age': 35,
#             'SibSp': 1,
#             'Parch': 0,
#             'Fare': 80.0,
#             'Embarked': 'S',
#             'Cabin': 'C85',
#             'Ticket': 'PC 17599',
#             'Name': 'Smith, Mrs. John (Mary)'
#         },
#         'third_class_man': {
#             'Pclass': 3,
#             'Sex': 'male',
#             'Age': 25, 
#             'SibSp': 0,
#             'Parch': 0,
#             'Fare': 8.0,
#             'Embarked': 'S',
#             'Cabin': np.nan,
#             'Ticket': '12345',
#             'Name': 'Jones, Mr. William'
#         },
#         'child_with_family': {
#             'Pclass': 2,
#             'Sex': 'female',
#             'Age': 8,
#             'SibSp': 1,
#             'Parch': 2, 
#             'Fare': 25.0,
#             'Embarked': 'C',
#             'Cabin': np.nan,
#             'Ticket': '54321', 
#             'Name': 'Wilson, Miss. Emma'
#         }
#     }
    
#     return examples
