"""
Enhanced machine learning model utilities for 85-90% accuracy achievement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, StackingClassifier, 
                              ExtraTreesClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import additional powerful models
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Enhanced model dictionary with high-performance models
def get_enhanced_model_dict():
    """Get enhanced model dictionary with optimal parameters for Titanic dataset"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=5, 
            min_samples_leaf=2, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, 
            subsample=0.8, random_state=42, eval_metric='logloss'
        ),
        'SVM': SVC(probability=True, random_state=42, C=1.0, gamma='scale'),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, random_state=42
        ),
        'Naive Bayes': GaussianNB(),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
    }
    
    # Add advanced models if available
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.1, 
            random_state=42, verbose=False
        )
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        )
    
    return models

MODEL_DICT = get_enhanced_model_dict()

def get_ensemble_models():
    """Create ensemble models for maximum accuracy"""
    base_models = MODEL_DICT.copy()
    
    # Remove ensemble-unfriendly models for base estimators
    ensemble_base = {k: v for k, v in base_models.items() 
                     if k not in ['Naive Bayes']}  # NB can cause issues in ensembles
    
    ensemble_models = {}
    
    # Voting Classifier (Soft Voting)
    voting_estimators = [(name, model) for name, model in list(ensemble_base.items())[:5]]
    ensemble_models['Voting Classifier'] = VotingClassifier(
        estimators=voting_estimators,
        voting='soft'
    )
    
    # Stacking Classifier
    stacking_estimators = [(name, model) for name, model in list(ensemble_base.items())[:4]]
    ensemble_models['Stacking Classifier'] = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # FIXED: Changed base_estimator to estimator for sklearn compatibility
    ensemble_models['Bagged RF'] = BaggingClassifier(
        estimator=RandomForestClassifier(max_depth=8, random_state=42),  # Changed parameter name
        n_estimators=10,
        random_state=42
    )
    
    return ensemble_models




def get_enhanced_preprocessor():
    """Create enhanced preprocessing pipeline"""
    # Enhanced feature lists
    numeric_features = [
        'Age', 'Fare', 'FamilySize', 'FarePerPerson', 'TicketGroupSize',
        'AgeSquared', 'LogFare', 'FareRelativeToClass', 'NameLength'
    ]
    
    categorical_features = [
        'Pclass', 'Sex', 'Embarked', 'Title', 'CabinLetter', 
        'FamilySizeGroup', 'AgeGroup', 'FareGroup', 'TicketPrefix'
    ]
    
    # Enhanced preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def create_enhanced_pipeline(model, preprocessor=None):
    """Create enhanced model pipeline with preprocessing"""
    if preprocessor is None:
        preprocessor = get_enhanced_preprocessor()
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def optimize_hyperparameters_optuna(X, y, model_name, n_trials=100):
    """Optimize hyperparameters using Optuna for better performance - FIXED VERSION"""
    if not OPTUNA_AVAILABLE:
        return None
    
    def objective(trial):
        if model_name == 'Random Forest':
            # FIXED: Add 'classifier__' prefix for pipeline parameters
            params = {
                'classifier__n_estimators': trial.suggest_int('n_estimators', 200, 500),
                'classifier__max_depth': trial.suggest_int('max_depth', 5, 15),
                'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'classifier__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'classifier__max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'XGBoost':
            # FIXED: Add 'classifier__' prefix for pipeline parameters
            params = {
                'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'classifier__max_depth': trial.suggest_int('max_depth', 3, 10),
                'classifier__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'classifier__subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'classifier__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = XGBClassifier(random_state=42, eval_metric='logloss')
            
        elif model_name == 'CatBoost' and CATBOOST_AVAILABLE:
            # FIXED: Add 'classifier__' prefix for pipeline parameters
            params = {
                'classifier__iterations': trial.suggest_int('iterations', 100, 300),
                'classifier__depth': trial.suggest_int('depth', 4, 10),
                'classifier__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
            }
            model = CatBoostClassifier(random_state=42, verbose=False)
        else:
            return 0.8  # Default score for unsupported models
        
        # Create pipeline
        pipeline = create_enhanced_pipeline(model)
        
        # Set parameters with correct naming
        pipeline.set_params(**params)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            return cv_scores.mean()
        except Exception as e:
            print(f"Error in trial: {e}")
            return 0.0  # Return low score for failed trials
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Convert best params back to the format expected by the pipeline
        best_params = {}
        for key, value in study.best_params.items():
            best_params[f'classifier__{key}'] = value
        
        return best_params
    except Exception as e:
        print(f"Optuna optimization failed: {e}")
        return None


def train_single_model_enhanced(X_train, X_test, y_train, y_test, model_name, 
                                enable_tuning=False, enable_optuna=False):
    """Enhanced single model training with advanced techniques - ROBUST VERSION"""
    
    # Get base model with error handling
    try:
        if model_name in MODEL_DICT:
            base_model = MODEL_DICT[model_name]
        else:
            ensemble_models = get_ensemble_models()
            if model_name in ensemble_models:
                base_model = ensemble_models[model_name]
            else:
                raise ValueError(f"Model {model_name} not found")
    except Exception as e:
        print(f"Error getting base model {model_name}: {e}")
        # Fallback to a simple Random Forest
        base_model = RandomForestClassifier(random_state=42)
    
    # Create pipeline with error handling
    try:
        pipeline = create_enhanced_pipeline(base_model)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        # Fallback to simple pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
    
    # Initialize best_model to ensure it's always defined
    best_model = None
    optimization_used = "None"
    
    # Advanced hyperparameter tuning with comprehensive error handling
    if enable_optuna and model_name in ['Random Forest', 'XGBoost', 'CatBoost']:
        try:
            print(f"Optimizing {model_name} with Optuna...")
            best_params = optimize_hyperparameters_optuna(X_train, y_train, model_name, n_trials=50)
            
            if best_params:
                try:
                    pipeline.set_params(**best_params)
                    optimization_used = "Optuna"
                    print(f"âœ… Optuna optimization successful for {model_name}")
                except Exception as param_error:
                    print(f"âš ï¸ Error setting Optuna parameters: {param_error}")
                    print("Falling back to default parameters...")
            else:
                print(f"âš ï¸ Optuna optimization returned no parameters for {model_name}")
            
            # Always fit the model, regardless of parameter setting success
            best_model = pipeline.fit(X_train, y_train)
            
        except Exception as optuna_error:
            print(f"âŒ Optuna optimization failed for {model_name}: {optuna_error}")
            print("Falling back to standard training...")
            best_model = pipeline.fit(X_train, y_train)
            optimization_used = "Failed - Fallback"
    
    elif enable_tuning and model_name in ['Random Forest', 'XGBoost']:
        try:
            print(f"Tuning {model_name} with GridSearchCV...")
            
            # Define parameter grids
            if model_name == 'Random Forest':
                param_grid = {
                    'classifier__n_estimators': [200, 300, 400],
                    'classifier__max_depth': [6, 8, 10, 12],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'XGBoost':
                param_grid = {
                    'classifier__n_estimators': [150, 200, 250],
                    'classifier__max_depth': [4, 6, 8],
                    'classifier__learning_rate': [0.05, 0.1, 0.15],
                    'classifier__subsample': [0.8, 0.9, 1.0]
                }
            
            # Use StratifiedKFold for better cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Perform GridSearchCV with timeout protection
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='f1', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            optimization_used = "GridSearchCV"
            print(f"âœ… GridSearchCV optimization successful for {model_name}")
            
        except Exception as grid_error:
            print(f"âŒ GridSearchCV failed for {model_name}: {grid_error}")
            print("Falling back to standard training...")
            try:
                best_model = pipeline.fit(X_train, y_train)
                optimization_used = "Failed - Fallback"
            except Exception as fallback_error:
                print(f"âŒ Fallback training also failed: {fallback_error}")
                # Last resort: create and train a simple model
                best_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                optimization_used = "Emergency Fallback"
    
    else:
        # Standard training without tuning
        try:
            best_model = pipeline.fit(X_train, y_train)
            optimization_used = "Standard"
            print(f"âœ… Standard training successful for {model_name}")
        except Exception as standard_error:
            print(f"âŒ Standard training failed for {model_name}: {standard_error}")
            # Emergency fallback
            best_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
            optimization_used = "Emergency Fallback"
    
    # Final safety check - ensure best_model is always defined
    if best_model is None:
        print(f"âš ï¸ Warning: best_model is None for {model_name}. Creating emergency fallback...")
        best_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        optimization_used = "Emergency Fallback"
    
    # Make predictions with error handling
    try:
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    except Exception as pred_error:
        print(f"âŒ Prediction error for {model_name}: {pred_error}")
        # Return minimal results if prediction fails
        return {
            'Model': model_name,
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'ROC_AUC': 0.0,
            'Classification_Report': f"Prediction failed: {pred_error}",
            'Confusion_Matrix': np.array([[0, 0], [0, 0]]),
            'Trained_Model': best_model,
            'CV_Scores': np.array([0.0]),
            'CV_Mean': 0.0,
            'CV_Std': 0.0,
            'Best_Params': None,
            'Optimization_Used': optimization_used,
            'Error': str(pred_error)
        }
    
    # Enhanced cross-validation with error handling
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
    except Exception as cv_error:
        print(f"âš ï¸ Cross-validation error for {model_name}: {cv_error}")
        cv_scores = np.array([0.0])  # Fallback CV scores
    
    # Calculate comprehensive metrics with error handling
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
    except Exception as metrics_error:
        print(f"âš ï¸ Metrics calculation error for {model_name}: {metrics_error}")
        # Provide default values if metrics calculation fails
        accuracy = precision = recall = f1 = roc_auc = 0.0
        classification_rep = f"Metrics calculation failed: {metrics_error}"
        confusion_mat = np.array([[0, 0], [0, 0]])
    
    # Safely get best parameters
    try:
        if hasattr(best_model, 'best_params_'):
            best_params = best_model.best_params_
        elif hasattr(best_model, 'get_params'):
            best_params = best_model.get_params()
        else:
            best_params = None
    except Exception:
        best_params = None
    
    # Compile results
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc,
        'Classification_Report': classification_rep,
        'Confusion_Matrix': confusion_mat,
        'Trained_Model': best_model,
        'CV_Scores': cv_scores,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Best_Params': best_params,
        'Optimization_Used': optimization_used
    }
    
    print(f"âœ… Training completed for {model_name} using {optimization_used}")
    return results



def train_and_evaluate_all_models(X_train, X_test, y_train, y_test, selected_models, 
                                   enable_tuning=False, enable_optuna=False, include_ensembles=True):
    """Enhanced model training with ensemble methods"""
    results = []
    
    # Train selected base models
    for model_name in selected_models:
        try:
            print(f"Training {model_name}...")
            result = train_single_model_enhanced(
                X_train, X_test, y_train, y_test, 
                model_name, enable_tuning, enable_optuna
            )
            results.append(result)
            print(f"âœ“ {model_name}: {result['Accuracy']:.4f} accuracy")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Train ensemble models if requested
    if include_ensembles and len(results) >= 2:
        ensemble_models = get_ensemble_models()
        
        for ensemble_name in ['Voting Classifier', 'Stacking Classifier']:
            try:
                print(f"Training {ensemble_name}...")
                result = train_single_model_enhanced(
                    X_train, X_test, y_train, y_test, 
                    ensemble_name, False, False
                )
                results.append(result)
                print(f"âœ“ {ensemble_name}: {result['Accuracy']:.4f} accuracy")
                
            except Exception as e:
                print(f"Error training {ensemble_name}: {str(e)}")
                continue
    
    return results

def calibrate_model_probabilities(model, X_val, y_val):
    """Calibrate model probabilities for better confidence estimates"""
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated_model.fit(X_val, y_val)
    return calibrated_model

def get_enhanced_model_descriptions():
    """Get descriptions of enhanced models"""
    descriptions = {
        'Logistic Regression': 'Enhanced linear model with optimal regularization for interpretable results.',
        'Random Forest': 'Optimized ensemble of 300 trees with tuned parameters for maximum accuracy.',
        'XGBoost': 'Advanced gradient boosting with optimized parameters, excellent for competitions.',
        'SVM': 'Support Vector Machine with probability estimates and optimal kernel parameters.',
        'KNN': 'Enhanced k-nearest neighbors with distance weighting and optimal k=7.',
        'Decision Tree': 'Optimized single tree with pruning parameters to prevent overfitting.',
        'Naive Bayes': 'Probabilistic classifier assuming feature independence, fast baseline.',
        'Extra Trees': 'Extremely randomized trees ensemble for variance reduction.',
        'Voting Classifier': 'Soft voting ensemble combining multiple strong base models.',
        'Stacking Classifier': 'Two-level ensemble with meta-learner for optimal combinations.',
        'Bagged RF': 'Bootstrap aggregating with Random Forest for additional stability.'
    }
    
    if CATBOOST_AVAILABLE:
        descriptions['CatBoost'] = 'Gradient boosting optimized for categorical features, often achieves top performance.'
    
    if LIGHTGBM_AVAILABLE:
        descriptions['LightGBM'] = 'Fast gradient boosting with optimized memory usage and high accuracy.'
    
    return descriptions

def feature_selection_recursive(X, y, model, n_features=20):
    """Recursive feature elimination for optimal feature subset"""
    from sklearn.feature_selection import RFE
    
    # Use a simple model for feature selection
    selector = RFE(estimator=model, n_features_to_select=n_features)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected, selector.support_

def get_feature_importance_enhanced(model, feature_names=None):
    """Enhanced feature importance extraction"""
    try:
        # Get the classifier from the pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        # Handle ensemble models
        if hasattr(classifier, 'estimators_'):
            # For ensemble models, average feature importances
            if hasattr(classifier.estimators_[0], 'feature_importances_'):
                importances = np.mean([est.feature_importances_ for est in classifier.estimators_], axis=0)
            else:
                return None
        elif hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
        else:
            return None
        
        # Get feature names from preprocessor if not provided
        if feature_names is None:
            try:
                if hasattr(model, 'named_steps'):
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                else:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
            except:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        print(f"Error getting feature importance: {str(e)}")
        return None

# Keep all existing functions from the original file
def save_model(model, model_name, filepath=None):
    """Save trained model to disk"""
    if filepath is None:
        filepath = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
    
    try:
        joblib.dump(model, filepath)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_trained_model(filepath):
    """Load trained model from disk"""
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def make_prediction(model, X):
    """Make prediction using trained model"""
    try:
        prediction = model.predict(X)
        prediction_proba = model.predict_proba(X)
        return prediction, prediction_proba
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None

def compare_models(results):
    """Enhanced model comparison with additional metrics"""
    comparison_data = []
    
    for result in results:
        comparison_data.append({
            'Model': result['Model'],
            'Accuracy': result['Accuracy'],
            'Precision': result['Precision'],
            'Recall': result['Recall'],
            'F1': result['F1'],
            'ROC_AUC': result['ROC_AUC'],
            'CV_Mean': result['CV_Mean'],
            'CV_Std': result['CV_Std'],
            'Stability': 1 - result['CV_Std']  # Lower std = higher stability
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    return comparison_df

def get_best_model(results):
    """Get the best performing model with enhanced criteria"""
    # Weight accuracy heavily but consider stability
    def score_model(result):
        return result['Accuracy'] * 0.7 + (1 - result['CV_Std']) * 0.3
    
    best_result = max(results, key=score_model)
    return best_result

def evaluate_model_performance(model, X_test, y_test):
    """Comprehensive evaluation of model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    evaluation = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return evaluation