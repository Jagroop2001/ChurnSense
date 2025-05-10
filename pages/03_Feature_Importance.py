import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.model_builder import get_feature_importance, get_shap_values
from utils.visualizations import plot_feature_importance, plot_shap_summary, SHAP_AVAILABLE

st.set_page_config(page_title="Feature Importance", page_icon="📈", layout="wide")

st.title("Feature Importance Analysis")

st.markdown("""
This section helps you understand which factors contribute most to customer churn.
It uses model-based feature importance and SHAP (SHapley Additive exPlanations) values to provide insights.
""")

# Check if data and models are available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data in the Data Upload section first.")
    st.stop()

if 'model_lr' not in st.session_state or st.session_state.model_lr is None or \
   'model_xgb' not in st.session_state or st.session_state.model_xgb is None:
    st.warning("Please train the models in the Churn Prediction section first.")
    st.stop()

# Feature importance analysis
st.subheader("Model-based Feature Importance")

# Let user select the model for feature importance
model_option = st.radio(
    "Select model for feature importance:",
    ["Logistic Regression", "XGBoost"],
    horizontal=True
)

# Calculate feature importance based on selected model
if model_option == "Logistic Regression":
    if 'feature_importance_lr' not in st.session_state:
        with st.spinner("Calculating feature importance for Logistic Regression..."):
            st.session_state.feature_importance_lr = get_feature_importance(
                st.session_state.model_lr,
                st.session_state.preprocessor,
                st.session_state.X_test
            )
    
    importance_df = st.session_state.feature_importance_lr
    
else:  # XGBoost
    if 'feature_importance_xgb' not in st.session_state:
        with st.spinner("Calculating feature importance for XGBoost..."):
            st.session_state.feature_importance_xgb = get_feature_importance(
                st.session_state.model_xgb,
                st.session_state.preprocessor,
                st.session_state.X_test
            )
    
    importance_df = st.session_state.feature_importance_xgb

# Plot feature importance
if importance_df is not None:
    st.write(f"Top features driving churn according to {model_option}:")
    
    # Let user select how many features to display
    top_n = st.slider("Number of top features to display:", 5, 30, 15)
    
    fig = plot_feature_importance(importance_df, top_n)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the feature importance table
    st.write("Feature Importance Table:")
    st.dataframe(importance_df.head(top_n))

# SHAP Analysis section
st.markdown("---")
st.subheader("SHAP Value Analysis")

if not SHAP_AVAILABLE:
    st.warning("The SHAP library is not installed. SHAP analysis features are disabled.")
    st.info("SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance and show how each feature contributes to pushing the prediction higher or lower.")
else:
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance.
    They show how each feature contributes to pushing the prediction higher or lower.
    """)

    # Calculate SHAP values
    if st.button("Calculate SHAP Values"):
        with st.spinner("Calculating SHAP values (this may take a moment)..."):
            try:
                # Usually XGBoost works better with SHAP
                shap_values, expected_value, feature_names = get_shap_values(
                    st.session_state.model_xgb,
                    st.session_state.preprocessor,
                    st.session_state.X_test
                )
                
                st.session_state.shap_values = shap_values
                st.session_state.shap_expected_value = expected_value
                st.session_state.shap_feature_names = feature_names
                
                if shap_values is None:
                    st.warning("SHAP values could not be calculated. See the fallback visualization below.")
                else:
                    st.success("SHAP values calculated successfully!")
                
            except Exception as e:
                st.error(f"Error calculating SHAP values: {str(e)}")

# Display SHAP summary plot if values are available
if 'shap_values' in st.session_state and st.session_state.shap_values is not None:
    st.write("SHAP Summary Plot:")
    
    # Number of features to display in SHAP summary
    max_display = st.slider("Number of features to display in SHAP summary:", 5, 30, 20)
    
    try:
        # Plot SHAP summary
        fig = plot_shap_summary(
            st.session_state.shap_values,
            st.session_state.shap_feature_names,
            max_display
        )
        st.pyplot(fig)
        
        st.markdown("""
        ### Interpreting SHAP Values
        
        - **Feature Importance**: Features are ordered by importance (top to bottom)
        - **Impact Direction**: Red points indicate higher feature values, blue points indicate lower values
        - **SHAP Value**: Points to the right increase churn probability, points to the left decrease it
        - **Value Distribution**: How feature values are distributed across the dataset
        """)
        
    except Exception as e:
        st.error(f"Error displaying SHAP plot: {str(e)}")

# Feature insights section
st.markdown("---")
st.subheader("Feature Insights")

# Define some common feature groups and their business implications
feature_insights = {
    "contract": "Contract type significantly impacts churn. Month-to-month contracts often show higher churn rates compared to longer-term contracts.",
    "tenure": "Customer tenure is typically inversely related to churn risk. Newer customers are more likely to churn than those who have been with the company longer.",
    "monthly": "Higher monthly charges often correlate with increased churn probability, suggesting price sensitivity.",
    "service": "Service quality issues or lack of service usage can be strong indicators of churn risk.",
    "support": "Interactions with customer support can signal satisfaction issues that lead to churn.",
    "payment": "Payment-related features (methods, automatic billing) can impact customer experience and churn.",
    "dependents": "Customers with dependents or family plans may have different churn patterns than individual subscribers."
}

# Display insights for detected features
if importance_df is not None:
    detected_insights = []
    
    for category, insight in feature_insights.items():
        # Check if any important feature contains this category
        matching_features = importance_df['Feature'].str.contains(category, case=False)
        if matching_features.any():
            detected_insights.append((category, insight))
    
    if detected_insights:
        st.write("Business Insights Based on Important Features:")
        
        for category, insight in detected_insights:
            st.markdown(f"**{category.capitalize()}**: {insight}")
    
    # Actionable recommendations based on top features
    st.markdown("### Recommended Actions")
    
    top_features = importance_df.head(5)['Feature'].tolist()
    
    # Generate some generic recommendations based on common features
    if any('contract' in feature.lower() for feature in top_features):
        st.markdown("🔹 **Offer Contract Incentives**: Encourage customers to switch to longer-term contracts with appropriate incentives.")
    
    if any('tenure' in feature.lower() for feature in top_features):
        st.markdown("🔹 **New Customer Onboarding**: Strengthen onboarding and early engagement for new customers who are at higher risk.")
    
    if any('monthly' in feature.lower() and 'charge' in feature.lower() for feature in top_features):
        st.markdown("🔹 **Pricing Tiers**: Evaluate price sensitivity and consider alternative pricing tiers or bundles.")
    
    if any('service' in feature.lower() for feature in top_features):
        st.markdown("🔹 **Service Utilization**: Promote key services or features that drive retention.")
    
    if any('support' in feature.lower() for feature in top_features):
        st.markdown("🔹 **Support Experience**: Improve customer support experiences, especially for high-risk segments.")
    
    # Generic recommendations
    st.markdown("🔹 **Targeted Intervention**: Develop personalized outreach campaigns for high-risk customers.")
    st.markdown("🔹 **Feedback Loop**: Collect and act on feedback from recently churned customers.")

# Show next steps
st.markdown("---")
st.markdown("""
### Next Steps

Now that you understand which factors contribute most to churn, you can:

1. Perform **Customer Segmentation** to identify different user groups with similar churn patterns
2. Analyze churn trends over time with **Cohort Analysis**
3. View **Retention Recommendations** tailored to your customer base
""")
