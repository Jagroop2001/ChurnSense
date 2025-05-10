import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import time

# Try to import shap, but continue if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not available. SHAP analysis features will be disabled.")

def train_logistic_regression(X_train, y_train, preprocessor, params=None):
    """
    Train a Logistic Regression model
    
    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training target variable
    preprocessor (ColumnTransformer): Data preprocessor
    params (dict): Parameters for LogisticRegression
    
    Returns:
    tuple: (model, training_time)
    """
    start_time = time.time()
    
    # Default parameters if none provided
    if params is None:
        params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
    
    # Create and train model
    model = LogisticRegression(**params)
    
    # Transform the data using the preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train the model
    model.fit(X_train_processed, y_train)
    
    training_time = time.time() - start_time
    
    return model, training_time

def train_xgboost(X_train, y_train, preprocessor, params=None):
    """
    Train an XGBoost model
    
    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training target variable
    preprocessor (ColumnTransformer): Data preprocessor
    params (dict): Parameters for XGBoost
    
    Returns:
    tuple: (model, training_time)
    """
    start_time = time.time()
    
    # Default parameters if none provided
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Transform the data using the preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train the model
    model.fit(X_train_processed, y_train)
    
    training_time = time.time() - start_time
    
    return model, training_time

def evaluate_model(model, X_test, y_test, preprocessor, model_name):
    """
    Evaluate a trained model on test data
    
    Parameters:
    model: Trained model
    X_test (DataFrame): Testing features
    y_test (Series): Testing target variable
    preprocessor (ColumnTransformer): Data preprocessor
    model_name (str): Name of the model for reporting
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # For probability scores (needed for AUC)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_pred_proba = y_pred  # Fallback if predict_proba not available
    
    # Calculate evaluation metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division='warn'),
        'recall': recall_score(y_test, y_pred, zero_division='warn'),
        'f1_score': f1_score(y_test, y_pred, zero_division='warn'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix and classification report for more detailed analysis
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics

def get_feature_importance(model, preprocessor, X):
    """
    Extract feature importance from a trained model
    
    Parameters:
    model: Trained model (LogisticRegression or XGBoost)
    preprocessor (ColumnTransformer): Data preprocessor
    X (DataFrame): Original feature dataframe (for column names)
    
    Returns:
    DataFrame: Feature importances
    """
    # Transform the data
    X_processed = preprocessor.transform(X)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Extract feature names from the column transformer
    # Get transformers, with the assumption that categorical features are one-hot encoded
    transformers = preprocessor.transformers_
    
    for name, transformer, columns in transformers:
        if name == 'num':  # For numeric features
            feature_names.extend(columns)
        elif name == 'cat':  # For categorical features
            # For one-hot encoded features, get the categories
            for i, col in enumerate(columns):
                # Get the feature name template
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out([col])
                    feature_names.extend(cat_features)
                else:
                    # Fallback for older scikit-learn versions
                    categories = transformer.named_steps['onehot'].categories_[i]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
    
    # Get feature importance based on model type
    if isinstance(model, LogisticRegression):
        # For logistic regression, we take the absolute coefficients
        importances = np.abs(model.coef_[0])
    elif isinstance(model, xgb.XGBClassifier):
        # For XGBoost, use the feature_importances_ attribute
        importances = model.feature_importances_
    else:
        raise ValueError("Unsupported model type for feature importance extraction")
    
    # Make sure the number of feature names matches the number of importances
    if len(feature_names) != len(importances):
        # Trim or pad feature names to match importances
        feature_names = feature_names[:len(importances)] if len(feature_names) > len(importances) else feature_names + [f"Feature_{i}" for i in range(len(feature_names), len(importances))]
    
    # Create a DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance in descending order
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def get_shap_values(model, preprocessor, X_test):
    """
    Calculate SHAP values for model explainability
    
    Parameters:
    model: Trained model (XGBoost or LogisticRegression)
    preprocessor (ColumnTransformer): Data preprocessor
    X_test (DataFrame): Test features
    
    Returns:
    tuple: (shap_values, expected_value, feature_names) or (None, None, feature_names) if SHAP is not available
    """
    # Check if SHAP is available
    if not SHAP_AVAILABLE:
        return None, None, None
        
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names (similar to get_feature_importance function)
    feature_names = []
    transformers = preprocessor.transformers_
    
    for name, transformer, columns in transformers:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            for i, col in enumerate(columns):
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out([col])
                    feature_names.extend(cat_features)
                else:
                    categories = transformer.named_steps['onehot'].categories_[i]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
    
    # Make sure feature_names length matches the processed data
    if len(feature_names) != X_test_processed.shape[1]:
        feature_names = feature_names[:X_test_processed.shape[1]] if len(feature_names) > X_test_processed.shape[1] else feature_names + [f"Feature_{i}" for i in range(len(feature_names), X_test_processed.shape[1])]
    
    try:
        # Calculate SHAP values based on model type
        if isinstance(model, xgb.XGBClassifier):
            # XGBoost-specific SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test_processed)
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]  # For binary classification, use the positive class
        else:
            # For other models like LogisticRegression
            explainer = shap.LinearExplainer(model, X_test_processed)
            shap_values = explainer.shap_values(X_test_processed)
            expected_value = explainer.expected_value
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
        
        return shap_values, expected_value, feature_names
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None, None, feature_names

def predict_churn_probability(model, preprocessor, X_data):
    """
    Predict churn probabilities for a dataset
    
    Parameters:
    model: Trained model
    preprocessor (ColumnTransformer): Data preprocessor
    X_data (DataFrame): Features to predict on
    
    Returns:
    array: Churn probabilities
    """
    # Transform the data
    X_processed = preprocessor.transform(X_data)
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_processed)[:, 1]
    else:
        # Fallback if predict_proba not available (rare, but possible)
        probabilities = model.predict(X_processed)
    
    return probabilities
