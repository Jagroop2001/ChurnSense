import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from utils.data_processor import load_and_validate_data, preprocess_data, analyze_data_quality, get_data_summary
from utils.visualizations import plot_categorical_distribution, plot_numeric_distribution, plot_correlation_heatmap

st.set_page_config(page_title="Data Upload & Overview", page_icon="📊", layout="wide")

st.title("Data Upload & Overview")

st.markdown("""
This section allows you to upload your customer data and provides an overview of its characteristics.
The data should be in CSV format and include customer attributes and a churn indicator column.
""")

# File uploader for CSV data
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file is not None:
    # Load and validate the uploaded data
    success, result = load_and_validate_data(uploaded_file)
    
    if success:
        # Store the data in session_state
        st.session_state.data = result
        
        # Display success message
        st.success("Data successfully loaded!")
        
        # Display basic information about the dataset
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Rows: {st.session_state.data.shape[0]}")
            st.write(f"Columns: {st.session_state.data.shape[1]}")
        
        with col2:
            # Try to identify the target column (churn indicator)
            churn_cols = [col for col in st.session_state.data.columns if 'churn' in col.lower()]
            if churn_cols:
                target_col = churn_cols[0]
                churn_count = st.session_state.data[target_col].sum()
                churn_rate = (churn_count / st.session_state.data.shape[0]) * 100
                st.write(f"Churn Column: {target_col}")
                st.write(f"Churn Rate: {churn_rate:.2f}%")
            else:
                st.warning("No obvious churn column detected. Please select one in the Churn Prediction tab.")
        
        # Display the first few rows of the dataset
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10))
        
        # Data summary section
        st.subheader("Data Summary")
        
        # Get data summary
        data_summary = get_data_summary(st.session_state.data)
        
        # Display summary statistics for numeric columns
        if 'numeric_summary' in data_summary:
            st.write("Numeric Columns Summary:")
            st.dataframe(data_summary['numeric_summary'])
        
        # Display summary for categorical columns
        if 'categorical_summary' in data_summary:
            st.write("Categorical Columns Summary:")
            for col, summary in data_summary['categorical_summary'].items():
                st.write(f"{col}: {summary['unique_values']} unique values")
                
                # Show top values in a horizontal bar chart
                top_values = pd.DataFrame(
                    list(summary['top_values'].items()), 
                    columns=[col, 'Count']
                ).sort_values('Count', ascending=False)
                
                fig = px.bar(
                    top_values, 
                    x='Count', 
                    y=col, 
                    orientation='h',
                    title=f"Top values for {col}",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data quality analysis
        st.subheader("Data Quality Analysis")
        
        quality_report = analyze_data_quality(st.session_state.data)
        
        # Missing values
        if not quality_report['missing_values'].empty:
            missing_df = quality_report['missing_values'][quality_report['missing_values']['Count'] > 0]
            if not missing_df.empty:
                st.write("Missing Values:")
                st.dataframe(missing_df)
                
                # Plot missing values percentage
                fig = px.bar(
                    missing_df.sort_values('Percentage', ascending=False),
                    x=missing_df.index,
                    y='Percentage',
                    title="Missing Values Percentage by Column",
                    labels={'x': 'Column', 'y': 'Missing %'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No missing values found in the dataset.")
        
        # Duplicates
        st.write(f"Duplicate Rows: {quality_report['duplicate_rows']} ({quality_report['duplicate_pct']:.2f}%)")
        
        # Distribution analysis
        st.subheader("Distribution Analysis")
        
        # Get column types
        numeric_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude target column from feature columns if it exists
        target_col = None
        if 'churn_cols' in locals() and churn_cols:
            target_col = churn_cols[0]
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
        
        # Let the user select columns to analyze
        st.write("Select columns to analyze:")
        
        # For categorical columns
        if categorical_cols:
            st.write("Categorical Features:")
            selected_cat = st.multiselect(
                "Select categorical columns:",
                categorical_cols,
                default=categorical_cols[:min(3, len(categorical_cols))]
            )
            
            for col in selected_cat:
                fig = plot_categorical_distribution(st.session_state.data, col, target_col)
                st.plotly_chart(fig, use_container_width=True)
        
        # For numeric columns
        if numeric_cols:
            st.write("Numeric Features:")
            selected_num = st.multiselect(
                "Select numeric columns:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            for col in selected_num:
                fig = plot_numeric_distribution(st.session_state.data, col, target_col)
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis for numeric features
        if len(numeric_cols) > 1:
            st.subheader("Correlation Analysis")
            st.write("Correlation between numeric features:")
            
            fig = plot_correlation_heatmap(st.session_state.data, numeric_cols)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Display error message if data validation failed
        st.error(f"Error loading data: {result}")

else:
    # If no file is uploaded, show sample dataset option
    st.info("Upload a CSV file to begin analysis, or use a sample dataset below.")
    
    # Option to use sample dataset
    if st.button("Use Sample Telco Customer Churn Dataset"):
        # IBM Telco Customer Churn dataset
        sample_data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        try:
            sample_data = pd.read_csv(sample_data_url)
            st.session_state.data = sample_data
            st.success("Sample dataset loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading sample dataset: {str(e)}")

# Show next steps if data is loaded
if 'data' in st.session_state and st.session_state.data is not None:
    st.markdown("---")
    st.markdown("""
    ### Next Steps
    
    Now that your data is loaded and analyzed, you can:
    
    1. Proceed to the **Churn Prediction** page to train models
    2. Explore **Feature Importance** to understand factors affecting churn
    3. Perform **Customer Segmentation** to identify different user groups
    """)
