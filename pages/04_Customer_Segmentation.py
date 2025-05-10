import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from utils.visualizations import plot_customer_segments

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("Customer Segmentation")

st.markdown("""
This section helps you identify distinct customer segments based on their attributes and behaviors.
Understanding different customer groups enables targeted retention strategies.
""")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data in the Data Upload section first.")
    st.stop()

# Customer segmentation section
st.subheader("Segment Customers by Key Attributes")

# Get the list of columns
all_columns = st.session_state.data.columns.tolist()

# Try to identify the target column (churn indicator)
churn_cols = [col for col in all_columns if 'churn' in col.lower()]
target_col = churn_cols[0] if churn_cols else None

# Let user select features for segmentation
st.write("Select features to use for customer segmentation:")
segmentation_features = st.multiselect(
    "Features for segmentation:",
    [col for col in all_columns if col != target_col],
    # Try to make intelligent default selections based on column names
    default=[col for col in all_columns if col != target_col and any(term in col.lower() for term in 
                                                                   ['tenure', 'monthly', 'charge', 'usage', 'spend', 'revenue'])][:5]
)

# Only proceed if at least 2 features are selected
if len(segmentation_features) < 2:
    st.warning("Please select at least 2 features for segmentation.")
else:
    # Get number of clusters
    num_clusters = st.slider("Number of segments (clusters):", 2, 10, 3)
    
    # Perform segmentation when requested
    if st.button("Perform Customer Segmentation"):
        with st.spinner("Segmenting customers..."):
            try:
                # Get data for selected features
                segment_data = st.session_state.data[segmentation_features].copy()
                
                # Handle categorical features - convert to one-hot encoding
                segment_data_encoded = pd.get_dummies(segment_data)
                
                # Handle missing values
                segment_data_encoded.fillna(segment_data_encoded.mean(), inplace=True)
                
                # Scale the data
                scaler = StandardScaler()
                segment_data_scaled = scaler.fit_transform(segment_data_encoded)
                
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(segment_data_scaled)
                
                # Add segment labels to the original data
                segment_result = st.session_state.data.copy()
                segment_result['Segment'] = clusters
                segment_result['Segment'] = 'Segment ' + (segment_result['Segment'] + 1).astype(str)
                
                # Store segmentation results in session state
                st.session_state.segments = segment_result
                st.session_state.segment_features = segmentation_features
                
                st.success("Customer segmentation completed!")
                
            except Exception as e:
                st.error(f"Error performing segmentation: {str(e)}")
    
    # Display segmentation results if available
    if 'segments' in st.session_state and st.session_state.segments is not None:
        st.markdown("---")
        st.subheader("Segmentation Results")
        
        # Get segment distribution
        segment_distribution = st.session_state.segments['Segment'].value_counts().reset_index()
        segment_distribution.columns = ['Segment', 'Count']
        
        # Display segment distribution
        fig = px.pie(
            segment_distribution, 
            values='Count', 
            names='Segment',
            title='Customer Segment Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate churn rate by segment if target column exists
        if target_col:
            try:
                # Handle the case where target column might be non-numeric
                segment_data = st.session_state.segments.copy()
                
                # If the target column is not numeric, try to convert it
                if segment_data[target_col].dtype == 'object':
                    # For binary string values like 'Yes'/'No', convert to 1/0
                    if set(segment_data[target_col].dropna().unique()).issubset({'Yes', 'No', 'yes', 'no', 'True', 'False', 'true', 'false'}):
                        segment_data[target_col] = segment_data[target_col].map(
                            {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
                        )
                    else:
                        st.warning(f"Cannot calculate churn rate - column '{target_col}' contains non-binary string values.")
                        # Skip the churn rate calculation for this column
                        target_col = None
                
                # Calculate churn rate by segment
                segment_churn = segment_data.groupby('Segment')[target_col].mean().reset_index()
                segment_churn.columns = ['Segment', 'Churn Rate']
                
                # Display churn rate by segment
                fig = px.bar(
                    segment_churn.sort_values('Churn Rate'), 
                    x='Segment', 
                    y='Churn Rate',
                    title='Churn Rate by Segment',
                    color='Churn Rate',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Error calculating segment churn rates: {str(e)}")
        
        # Segment profiles - average values for key features
        st.subheader("Segment Profiles")
        
        profile_features = st.multiselect(
            "Select features to profile segments:",
            [col for col in all_columns if col != 'Segment'],
            default=segmentation_features[:min(5, len(segmentation_features))]
        )
        
        if profile_features:
            # Calculate mean values by segment for selected features
            segment_profiles = st.session_state.segments.groupby('Segment')[profile_features].mean().reset_index()
            
            # Display segment profiles
            st.write("Average values by segment:")
            st.dataframe(segment_profiles)
            
            # Visualize segment profiles
            # Melt the dataframe for easier plotting
            profile_melted = segment_profiles.melt(
                id_vars='Segment', 
                value_vars=profile_features,
                var_name='Feature', 
                value_name='Value'
            )
            
            # Create a grouped bar chart
            fig = px.bar(
                profile_melted, 
                x='Feature', 
                y='Value', 
                color='Segment',
                barmode='group',
                title='Segment Profiles by Feature',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Option to see segment feature distributions
        st.subheader("Feature Distributions by Segment")
        
        # Select a feature to analyze by segment
        dist_feature = st.selectbox(
            "Select a feature to view its distribution across segments:",
            profile_features
        )
        
        if dist_feature:
            # Create histograms for the selected feature by segment
            fig = px.histogram(
                st.session_state.segments,
                x=dist_feature,
                color='Segment',
                marginal="box",
                title=f'Distribution of {dist_feature} by Segment',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 2D visualization of segments
        if len(segmentation_features) >= 2:
            st.subheader("2D Segment Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox("X-axis feature:", segmentation_features)
            
            with col2:
                # Filter out the feature already selected for X-axis
                remaining_features = [f for f in segmentation_features if f != x_feature]
                y_feature = st.selectbox("Y-axis feature:", remaining_features)
            
            # Create a scatter plot for the selected features colored by segment
            fig = px.scatter(
                st.session_state.segments,
                x=x_feature,
                y=y_feature,
                color='Segment',
                title=f'Customer Segments: {x_feature} vs {y_feature}',
                color_discrete_sequence=px.colors.qualitative.Set3,
                opacity=0.7
            )
            
            # Add segment centroids if using KMeans
            if target_col:
                fig.update_layout(
                    shapes=[
                        dict(
                            type="circle",
                            xref="x",
                            yref="y",
                            x0=x_min,
                            y0=y_min,
                            x1=x_max,
                            y1=y_max,
                            line_color="gray",
                            fillcolor="gray",
                            opacity=0.3
                        ) for x_min, y_min, x_max, y_max in zip(
                            st.session_state.segments.groupby('Segment')[x_feature].mean() - 0.5,
                            st.session_state.segments.groupby('Segment')[y_feature].mean() - 0.5,
                            st.session_state.segments.groupby('Segment')[x_feature].mean() + 0.5,
                            st.session_state.segments.groupby('Segment')[y_feature].mean() + 0.5
                        )
                    ]
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment recommendations
        st.markdown("---")
        st.subheader("Segment-Based Recommendations")
        
        # If we have churn information, provide targeted recommendations
        if target_col and 'Segment' in st.session_state.segments.columns:
            # Get churn rates by segment
            churn_by_segment = st.session_state.segments.groupby('Segment')[target_col].mean().sort_values(ascending=False)
            
            # Display recommendations for each segment, prioritizing high-churn segments
            for segment, churn_rate in churn_by_segment.items():
                with st.expander(f"{segment} - Churn Rate: {churn_rate:.2%}"):
                    # Get segment profile
                    profile = st.session_state.segments[st.session_state.segments['Segment'] == segment][profile_features].mean()
                    
                    st.write("Segment Profile:")
                    st.dataframe(pd.DataFrame(profile).T)
                    
                    # Generate recommendations based on segment characteristics
                    st.write("Retention Recommendations:")
                    
                    if churn_rate > 0.3:  # High churn segment
                        st.markdown("🔴 **High Risk Segment** - Requires immediate attention")
                        st.markdown("- Implement proactive outreach campaign")
                        st.markdown("- Offer targeted incentives or promotions")
                        st.markdown("- Conduct satisfaction surveys to identify pain points")
                    elif churn_rate > 0.15:  # Medium churn segment
                        st.markdown("🟠 **Medium Risk Segment** - Monitor closely")
                        st.markdown("- Focus on improving customer experience")
                        st.markdown("- Provide relevant feature education")
                        st.markdown("- Consider loyalty rewards for retention")
                    else:  # Low churn segment
                        st.markdown("🟢 **Low Risk Segment** - Maintain satisfaction")
                        st.markdown("- Develop upsell/cross-sell opportunities")
                        st.markdown("- Create referral programs to leverage satisfaction")
                        st.markdown("- Continue monitoring for early warning signs")
        
        else:
            st.info("Add a churn indicator column to your data to receive targeted retention recommendations for each segment.")

# Show next steps
st.markdown("---")
st.markdown("""
### Next Steps

Now that you've segmented your customers, you can:

1. Analyze churn patterns over time with **Cohort Analysis**
2. View overall **Retention Recommendations** based on your segmentation and model insights
3. Develop segment-specific retention campaigns
""")
