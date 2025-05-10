import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_and_validate_data(uploaded_file):
    """
    Load and validate the uploaded CSV file
    
    Parameters:
    uploaded_file (FileUploader): The uploaded CSV file
    
    Returns:
    tuple: (success_flag, data_or_error_message)
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Basic validation checks
        if df.shape[0] < 10:
            return False, "Dataset has too few rows (minimum 10 required)"
        
        # Check if there's a churn column (target variable)
        churn_column_candidates = [col for col in df.columns if 'churn' in col.lower()]
        if not churn_column_candidates:
            return False, "No churn column detected. Please ensure your dataset has a column indicating churn."
        
        return True, df
    except Exception as e:
        return False, f"Error loading data: {str(e)}"

def preprocess_data(df, target_column=None):
    """
    Preprocess the data for model training
    
    Parameters:
    df (DataFrame): Input data
    target_column (str): The name of the target column (churn indicator)
    
    Returns:
    tuple: (processed_df, feature_columns, preprocessor)
    """
    # Create a copy to avoid modifying the original data
    data = df.copy()
    
    # Automatically detect the target column if not provided
    if target_column is None:
        churn_candidates = [col for col in data.columns if 'churn' in col.lower()]
        if churn_candidates:
            target_column = churn_candidates[0]
        else:
            raise ValueError("No target column specified and no 'churn' column detected")
    
    # Ensure the target column is binary
    if data[target_column].nunique() > 2:
        raise ValueError(f"Target column '{target_column}' has more than two unique values")
    
    # Convert target to binary if it's not already
    if not set(data[target_column].unique()).issubset({0, 1}):
        # Try to interpret string values like 'Yes'/'No' or 'True'/'False'
        if set(data[target_column].unique()).issubset({'Yes', 'No'}) or \
           set(data[target_column].unique()).issubset({'yes', 'no'}):
            data[target_column] = data[target_column].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        elif set(data[target_column].unique()).issubset({'True', 'False'}) or \
             set(data[target_column].unique()).issubset({'true', 'false'}):
            data[target_column] = data[target_column].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
        else:
            # For other cases, convert the smaller value to 0 and larger to 1
            unique_vals = sorted(data[target_column].unique())
            data[target_column] = data[target_column].map({unique_vals[0]: 0, unique_vals[1]: 1})
    
    # Identify numeric and categorical columns
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from features if present
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Return the data, feature columns and preprocessor
    feature_columns = numeric_features + categorical_features
    
    return data, feature_columns, target_column, preprocessor

def prepare_train_test_split(df, feature_columns, target_column, test_size=0.25, random_state=42):
    """
    Split the data into training and testing sets
    
    Parameters:
    df (DataFrame): Input data
    feature_columns (list): Columns to use as features
    target_column (str): The column containing the target variable
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def analyze_data_quality(df):
    """
    Analyze data quality issues like missing values, outliers, etc.
    
    Parameters:
    df (DataFrame): Input data
    
    Returns:
    dict: Dictionary containing data quality metrics
    """
    quality_report = {}
    
    # Missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    quality_report['missing_values'] = pd.DataFrame({
        'Count': missing_values,
        'Percentage': missing_pct
    })
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    quality_report['duplicate_rows'] = duplicate_rows
    quality_report['duplicate_pct'] = (duplicate_rows / len(df)) * 100
    
    # Data types
    quality_report['data_types'] = df.dtypes
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    quality_report['numeric_stats'] = df[numeric_cols].describe()
    
    # Value counts for categorical columns (top 5 values)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    value_counts = {}
    for col in categorical_cols:
        value_counts[col] = df[col].value_counts().head(5)
    quality_report['categorical_value_counts'] = value_counts
    
    return quality_report

def get_data_summary(df):
    """
    Generate a summary of the dataset
    
    Parameters:
    df (DataFrame): Input data
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {}
    
    # Basic information
    summary['shape'] = df.shape
    summary['columns'] = df.columns.tolist()
    summary['dtypes'] = df.dtypes.to_dict()
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_cols.empty:
        summary['numeric_summary'] = numeric_cols.describe()
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category'])
    if not categorical_cols.empty:
        cat_summary = {}
        for col in categorical_cols:
            cat_summary[col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        summary['categorical_summary'] = cat_summary
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary['missing_values'] = missing[missing > 0].to_dict()
    
    return summary
