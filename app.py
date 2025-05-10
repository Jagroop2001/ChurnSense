import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Churn Prediction & Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load an image from URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Create session states if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_lr' not in st.session_state:
    st.session_state.model_lr = None
if 'model_xgb' not in st.session_state:
    st.session_state.model_xgb = None
if 'predictions_lr' not in st.session_state:
    st.session_state.predictions_lr = None
if 'predictions_xgb' not in st.session_state:
    st.session_state.predictions_xgb = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'segments' not in st.session_state:
    st.session_state.segments = None

# Main page layout
st.title("Customer Churn Prediction & Analysis Platform")

# Dashboard header with image
col1, col2 = st.columns([1, 2])
with col1:
    # Use the analytics dashboard image
    image_url = "https://pixabay.com/get/g668cee00532eac033f4dbbf4d51454ba65cc461f5f830bcd4e14cbb4ca80d31598e1e661914c705999fa7535e7f755485a9a3f91cd1b6a5bf93f28d8f5995ebf_1280.jpg"
    try:
        img = load_image_from_url(image_url)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

with col2:
    st.markdown("""
    ## Welcome to the Churn Prediction & Analysis Platform
    
    This platform helps you:
    
    - **Predict** which customers are at risk of churning
    - **Understand** the key factors contributing to churn
    - **Segment** customers based on behavior and risk
    - **Analyze** churn patterns across different cohorts
    - **Develop** proactive customer retention strategies
    
    Get started by uploading your customer data in the Data Upload section.
    """)

# Main page dashboard overview
st.markdown("---")
st.header("Platform Overview")

# Cards for main features
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Data Analysis")
    st.markdown("""
    - Upload and preprocess your data
    - View data statistics and distributions
    - Identify data quality issues
    """)

with col2:
    st.subheader("🔮 Prediction Models")
    st.markdown("""
    - Train Logistic Regression model
    - Deploy XGBoost for advanced predictions
    - Evaluate model performance
    """)

with col3:
    st.subheader("🔍 Insights & Actions")
    st.markdown("""
    - Visualize feature importance
    - Segment customers by risk level
    - Generate retention recommendations
    """)

st.markdown("---")

# Show business impact section with image
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💼 Business Impact")
    st.markdown("""
    ### Why focus on churn prediction?
    
    - **Cost Efficiency**: Acquiring new customers costs 5-25x more than retaining existing ones
    - **Revenue Protection**: Reducing churn by just 5% can increase profits by 25-95%
    - **Proactive Strategy**: Address customer concerns before they lead to cancellations
    - **Resource Optimization**: Focus retention efforts on high-risk, high-value customers
    """)

with col2:
    # Use the business team meeting image
    image_url = "https://pixabay.com/get/gd7b70afde2ec1ce0c32f0ff53197e6e572c3e714650d7f464b4b267bd19aa2bb87b6c799b4f9a6f00e6cf8d4e422742b7162968ca47a039ae1b3db01ef872b8a_1280.jpg"
    try:
        img = load_image_from_url(image_url)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

# Customer retention concept
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    # Use customer retention concept image
    image_url = "https://pixabay.com/get/g9a7b0ac53dc963d6c2fb29513d3f7f04d5895a018c3d7cd96e519325ed3bdc39f0e56300d6ee5db9594d9e8d8ee20854cc2ec104105b8d6107eb22eff4b17876_1280.jpg"
    try:
        img = load_image_from_url(image_url)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

with col2:
    st.subheader("🤝 Customer Retention Strategy")
    st.markdown("""
    ### Turn predictions into action
    
    This platform helps you move from reactive to proactive customer retention:
    
    1. **Identify** at-risk customers before they churn
    2. **Understand** specific factors driving churn for different segments
    3. **Develop** targeted interventions based on customer needs
    4. **Measure** the effectiveness of your retention initiatives
    5. **Refine** your approach based on real results
    """)

# Footer with navigation guidance
st.markdown("---")
st.info("👈 Navigate through the different sections using the sidebar to explore all features.")
