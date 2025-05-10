import streamlit as st
import pandas as pd
import numpy as np
import time
from utils.data_processor import preprocess_data, prepare_train_test_split
from utils.model_builder import train_logistic_regression, train_xgboost, evaluate_model, predict_churn_probability
from utils.visualizations import plot_confusion_matrix, plot_model_metrics_comparison, plot_churn_prediction_distribution

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

st.title("Churn Prediction Models")

st.markdown("""
This section allows you to build and evaluate machine learning models to predict customer churn.
You can train both Logistic Regression and XGBoost models and compare their performance.
""")

# Check if data is loaded
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data in the Data Upload section first.")
    st.stop()

# Preprocessing and model training section
st.subheader("Model Configuration")

# Get the list of columns for target selection
all_columns = st.session_state.data.columns.tolist()

# Try to automatically identify the churn column
churn_column_candidates = [col for col in all_columns if 'churn' in col.lower()]
default_target = churn_column_candidates[0] if churn_column_candidates else all_columns[0]

# Let the user select the target column
target_column = st.selectbox(
    "Select the churn indicator column:",
    all_columns,
    index=all_columns.index(default_target) if default_target in all_columns else 0
)

# Display the class distribution of the target column
if target_column:
    target_counts = st.session_state.data[target_column].value_counts()
    st.write("Target Column Distribution:")
    
    # Create a horizontal bar chart for the target distribution
    target_df = pd.DataFrame({
        'Class': target_counts.index.astype(str),
        'Count': target_counts.values
    })
    
    fig = st.bar_chart(target_df.set_index('Class'))

# Model training section
st.subheader("Model Training")

col1, col2 = st.columns(2)

with col1:
    # Logistic Regression parameters
    st.write("Logistic Regression Parameters:")
    lr_c = st.slider("Regularization (C):", 0.01, 10.0, 1.0, 0.01)
    lr_solver = st.selectbox(
        "Solver:",
        options=['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        index=0
    )
    lr_max_iter = st.slider("Max Iterations:", 100, 1000, 500, 100)

with col2:
    # XGBoost parameters
    st.write("XGBoost Parameters:")
    xgb_n_estimators = st.slider("Number of Trees:", 50, 500, 100, 10)
    xgb_max_depth = st.slider("Max Tree Depth:", 1, 10, 3, 1)
    xgb_learning_rate = st.slider("Learning Rate:", 0.01, 0.3, 0.1, 0.01)

# Test size parameter
test_size = st.slider("Test Set Size:", 0.1, 0.5, 0.25, 0.05)

# Train the models button
if st.button("Train Models"):
    # Show a spinner during preprocessing and training
    with st.spinner("Preprocessing data and training models..."):
        try:
            # Preprocess the data
            processed_data, feature_columns, target_name, preprocessor = preprocess_data(
                st.session_state.data, target_column
            )
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = prepare_train_test_split(
                processed_data, feature_columns, target_name, test_size
            )
            
            # Store the split data for later use
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor
            st.session_state.feature_columns = feature_columns
            st.session_state.target_name = target_name
            
            # Set up parameters for Logistic Regression
            lr_params = {
                'C': lr_c,
                'solver': lr_solver,
                'max_iter': lr_max_iter,
                'random_state': 42
            }
            
            # Set up parameters for XGBoost
            xgb_params = {
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'learning_rate': xgb_learning_rate,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42
            }
            
            # Train Logistic Regression model
            start_time = time.time()
            lr_model, lr_train_time = train_logistic_regression(
                X_train, y_train, preprocessor, lr_params
            )
            st.session_state.model_lr = lr_model
            
            # Train XGBoost model
            xgb_model, xgb_train_time = train_xgboost(
                X_train, y_train, preprocessor, xgb_params
            )
            st.session_state.model_xgb = xgb_model
            
            # Evaluate models
            lr_metrics = evaluate_model(lr_model, X_test, y_test, preprocessor, "Logistic Regression")
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test, preprocessor, "XGBoost")
            
            # Store metrics in session state
            st.session_state.lr_metrics = lr_metrics
            st.session_state.xgb_metrics = xgb_metrics
            
            # Generate predictions for both models
            st.session_state.predictions_lr = predict_churn_probability(lr_model, preprocessor, X_test)
            st.session_state.predictions_xgb = predict_churn_probability(xgb_model, preprocessor, X_test)
            
            st.success("Models trained successfully!")
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")

# Display model results if models have been trained
if 'model_lr' in st.session_state and st.session_state.model_lr is not None and \
   'model_xgb' in st.session_state and st.session_state.model_xgb is not None:
    
    st.markdown("---")
    st.subheader("Model Results")
    
    # Display metrics comparison
    metrics_list = [st.session_state.lr_metrics, st.session_state.xgb_metrics]
    fig = plot_model_metrics_comparison(metrics_list)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display confusion matrices
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Logistic Regression Confusion Matrix:")
        fig_cm_lr = plot_confusion_matrix(st.session_state.lr_metrics['confusion_matrix'])
        st.plotly_chart(fig_cm_lr, use_container_width=True)
    
    with col2:
        st.write("XGBoost Confusion Matrix:")
        fig_cm_xgb = plot_confusion_matrix(st.session_state.xgb_metrics['confusion_matrix'])
        st.plotly_chart(fig_cm_xgb, use_container_width=True)
    
    # Show prediction distributions
    st.subheader("Prediction Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Logistic Regression Predictions:")
        fig_dist_lr = plot_churn_prediction_distribution(st.session_state.predictions_lr)
        st.plotly_chart(fig_dist_lr, use_container_width=True)
    
    with col2:
        st.write("XGBoost Predictions:")
        fig_dist_xgb = plot_churn_prediction_distribution(st.session_state.predictions_xgb)
        st.plotly_chart(fig_dist_xgb, use_container_width=True)
    
    # Show detailed metrics
    st.subheader("Detailed Model Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Logistic Regression:")
        lr_report = pd.DataFrame(st.session_state.lr_metrics['classification_report']).transpose()
        st.dataframe(lr_report)
    
    with col2:
        st.write("XGBoost:")
        xgb_report = pd.DataFrame(st.session_state.xgb_metrics['classification_report']).transpose()
        st.dataframe(xgb_report)

# Show next steps
if 'model_lr' in st.session_state and st.session_state.model_lr is not None and \
   'model_xgb' in st.session_state and st.session_state.model_xgb is not None:
    
    st.markdown("---")
    st.markdown("""
    ### Next Steps
    
    Now that you have trained the churn prediction models, you can:
    
    1. Explore **Feature Importance** to understand which factors contribute most to churn
    2. Perform **Customer Segmentation** to identify different customer groups and their churn risk
    3. Analyze **Cohort Analysis** to see how churn varies across different customer cohorts
    4. View **Retention Recommendations** based on the model findings
    """)
