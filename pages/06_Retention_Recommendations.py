import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Retention Recommendations", page_icon="🤝", layout="wide")

st.title("Customer Retention Recommendations")

# Function to load an image from URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Check if essential elements exist
has_data = 'data' in st.session_state and st.session_state.data is not None
has_models = 'model_lr' in st.session_state and st.session_state.model_lr is not None and \
             'model_xgb' in st.session_state and st.session_state.model_xgb is not None
has_features = 'feature_importance_xgb' in st.session_state or 'feature_importance_lr' in st.session_state
has_segments = 'segments' in st.session_state and st.session_state.segments is not None
has_cohorts = 'cohort_data' in st.session_state and st.session_state.cohort_data is not None

# Display appropriate header image
header_col1, header_col2 = st.columns([1, 2])

with header_col1:
    # Use customer retention concept image
    image_url = "https://pixabay.com/get/g94064f21f8cf1ed4b4c8d6886a58b5b532e419dbb561a0c93dd7ead8b3dce651ed6edb33305f19162ec1452101e43cc31f838a7722d4eaee169fae267d3c6628_1280.jpg"
    try:
        img = load_image_from_url(image_url)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

with header_col2:
    st.markdown("""
    ## Customer Retention Strategy
    
    This page provides actionable recommendations to reduce customer churn based on the insights gained from your data analysis.
    The recommendations are tailored to your specific customer base and business context.
    """)

# Display different content based on what analysis has been completed
if not has_data:
    st.warning("Please upload your data in the Data Upload section to receive retention recommendations.")
    st.stop()

# Get churn rate and customer count
data = st.session_state.data
churn_cols = [col for col in data.columns if 'churn' in col.lower()]
if churn_cols:
    target_col = churn_cols[0]
    total_customers = len(data)
    
    # Handle possible string values in churn column
    try:
        # Try to convert the column to numeric if it's not already
        if data[target_col].dtype == 'object':
            # For binary string values like 'Yes'/'No', convert to 1/0
            if set(data[target_col].dropna().unique()).issubset({'Yes', 'No', 'yes', 'no', 'True', 'False', 'true', 'false'}):
                churned_values = data[target_col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 
                                                       'True': 1, 'False': 0, 'true': 1, 'false': 0})
                churned_customers = churned_values.sum()
            else:
                # If it's some other string values, skip calculation
                churned_customers = "Unknown"
                churn_rate = "Unknown"
        else:
            # If it's already numeric, just sum it
            churned_customers = data[target_col].sum()
            
        # Calculate churn rate if churned_customers is numeric
        if isinstance(churned_customers, (int, float)):
            churn_rate = (churned_customers / total_customers) * 100
        else:
            churn_rate = "Unknown"
            
    except Exception as e:
        st.warning(f"Error calculating churn metrics: {e}")
        churned_customers = "Unknown"
        churn_rate = "Unknown"
else:
    # Default values if no churn column found
    total_customers = len(data)
    churn_rate = "Unknown"
    churned_customers = "Unknown"

# Display overall churn metrics
st.markdown("---")
st.subheader("Churn Overview")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Total Customers", f"{total_customers:,}")

with metric_col2:
    if isinstance(churned_customers, str):
        st.metric("Churned Customers", churned_customers)
    else:
        st.metric("Churned Customers", f"{int(churned_customers):,}")

with metric_col3:
    if isinstance(churn_rate, str):
        st.metric("Churn Rate", churn_rate)
    else:
        st.metric("Churn Rate", f"{churn_rate:.2f}%")

# Key insights section
st.markdown("---")
st.subheader("Key Insights & Recommendations")

# Different recommendations based on available analyses
if has_models and has_features:
    # Feature-based recommendations
    st.markdown("### ✨ Based on Feature Importance")
    
    # Get feature importance data
    if 'feature_importance_xgb' in st.session_state:
        importance_df = st.session_state.feature_importance_xgb
    else:
        importance_df = st.session_state.feature_importance_lr
    
    # Get top 5 features
    top_features = importance_df.head(5)['Feature'].tolist()
    
    # Display feature importance chart
    fig = px.bar(
        importance_df.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Factors Affecting Churn',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate recommendations based on common feature themes
    recommendations = []
    
    # Check for feature patterns in top features
    if any('contract' in feature.lower() for feature in top_features):
        recommendations.append({
            'title': 'Contract Optimization',
            'description': 'Customers with month-to-month contracts show higher churn rates',
            'actions': [
                'Offer incentives for longer-term contracts',
                'Create contract upgrade campaigns for month-to-month customers',
                'Highlight the benefits of longer commitments (price stability, added features)'
            ]
        })
    
    if any('tenure' in feature.lower() for feature in top_features):
        recommendations.append({
            'title': 'New Customer Onboarding',
            'description': 'Customers within their first 6-12 months show higher churn risk',
            'actions': [
                'Strengthen onboarding processes for new customers',
                'Implement regular check-ins during the first 6 months',
                'Provide usage tips and best practices to drive engagement'
            ]
        })
    
    if any(('price' in feature.lower() or 'charge' in feature.lower() or 'cost' in feature.lower()) for feature in top_features):
        recommendations.append({
            'title': 'Pricing Strategy Review',
            'description': 'Price sensitivity is a significant churn factor',
            'actions': [
                'Review competitive pricing strategies',
                'Implement value-based pricing tiers',
                'Offer loyalty discounts for long-term customers',
                'Consider price-lock guarantees for committed customers'
            ]
        })
    
    if any(('service' in feature.lower() or 'support' in feature.lower() or 'ticket' in feature.lower()) for feature in top_features):
        recommendations.append({
            'title': 'Service Quality Improvements',
            'description': 'Service issues are strongly correlated with churn',
            'actions': [
                'Review and enhance customer support processes',
                'Implement proactive support outreach for at-risk customers',
                'Develop service quality metrics and improvement targets',
                'Create recovery processes for customers who experience issues'
            ]
        })
    
    if any(('use' in feature.lower() or 'usage' in feature.lower() or 'activity' in feature.lower()) for feature in top_features):
        recommendations.append({
            'title': 'Engagement Enhancement',
            'description': 'Low product usage is a precursor to churn',
            'actions': [
                'Develop re-engagement campaigns for low-activity users',
                'Create targeted feature education campaigns',
                'Implement usage-based alerts to identify at-risk accounts',
                'Develop user journey maps to ensure feature discovery'
            ]
        })
    
    # Add generic recommendations if specific ones weren't identified
    if len(recommendations) < 3:
        generic_recommendations = [
            {
                'title': 'Customer Feedback Loop',
                'description': 'Systematic feedback collection improves retention',
                'actions': [
                    'Implement regular NPS or CSAT surveys',
                    'Create a structured process for acting on feedback',
                    'Follow up with detractors before they churn',
                    'Use customer feedback to prioritize product development'
                ]
            },
            {
                'title': 'Value Demonstration',
                'description': 'Regularly reinforce the value delivered to customers',
                'actions': [
                    'Create quarterly business reviews for key accounts',
                    'Send personalized usage and value reports',
                    'Highlight ROI and success metrics',
                    'Share case studies and success stories'
                ]
            },
            {
                'title': 'Proactive Risk Monitoring',
                'description': 'Identify churn risk before customers decide to leave',
                'actions': [
                    'Implement an early warning system using predictive models',
                    'Create a dedicated retention team for high-risk accounts',
                    'Develop escalation protocols for at-risk customers',
                    'Set up automated alerts for customer health changes'
                ]
            }
        ]
        
        # Add generic recommendations until we have at least 3
        for rec in generic_recommendations:
            if len(recommendations) < 3:
                recommendations.append(rec)
            else:
                break
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        with st.expander(f"{i+1}. {rec['title']} - {rec['description']}", expanded=True):
            st.markdown("**Recommended Actions:**")
            for action in rec['actions']:
                st.markdown(f"- {action}")

if has_segments:
    # Segment-based recommendations
    st.markdown("---")
    st.markdown("### 👥 Based on Customer Segmentation")
    
    # Get segment data
    segments = st.session_state.segments
    segment_column = 'Segment'
    
    # Check if we have churn information
    if churn_cols:
        target_col = churn_cols[0]
        
        # Calculate churn rate by segment
        segment_churn = segments.groupby(segment_column)[target_col].mean().sort_values(ascending=False)
        segment_counts = segments[segment_column].value_counts()
        
        # Create a segment summary table
        segment_summary = pd.DataFrame({
            'Size': segment_counts,
            'Proportion': segment_counts / segment_counts.sum(),
            'Churn Rate': segment_churn
        })
        
        # Display segment summary
        st.write("Segment Summary:")
        st.dataframe(segment_summary.style.format({
            'Size': '{:,.0f}',
            'Proportion': '{:.1%}',
            'Churn Rate': '{:.1%}'
        }))
        
        # Create a bubble chart showing segment size and churn rate
        segment_viz = pd.DataFrame({
            'Segment': segment_churn.index,
            'Churn Rate': segment_churn.values,
            'Size': segment_counts.values
        })
        
        fig = px.scatter(
            segment_viz,
            x='Segment',
            y='Churn Rate',
            size='Size',
            color='Churn Rate',
            hover_name='Segment',
            color_continuous_scale='Reds',
            title='Segment Size and Churn Rate',
            labels={'Churn Rate': 'Churn Rate'}
        )
        
        fig.update_layout(
            yaxis=dict(tickformat='.0%'),
            xaxis_title='Customer Segment',
            yaxis_title='Churn Rate'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment-specific recommendations
        st.markdown("#### Segment-Specific Strategies")
        
        # High-risk segments (top 2 by churn rate)
        high_risk_segments = segment_churn.head(2).index.tolist()
        
        # For high-risk segments
        st.markdown("**High-Risk Segments:**")
        for segment in high_risk_segments:
            churn_rate = segment_churn.loc[segment]
            segment_size = segment_counts.loc[segment]
            segment_pct = segment_size / segment_counts.sum()
            
            with st.expander(f"{segment} - Churn Rate: {churn_rate:.1%}, Size: {segment_size:,.0f} ({segment_pct:.1%} of customers)", expanded=True):
                st.markdown("**Priority Actions:**")
                
                # Generate recommendations based on churn rate
                if churn_rate > 0.4:
                    st.markdown("- 🔴 **Critical Priority**: Immediate retention campaign required")
                    st.markdown("- Conduct exit surveys to identify specific pain points")
                    st.markdown("- Offer targeted discounts or service upgrades")
                    st.markdown("- Assign dedicated account managers to high-value customers")
                    st.markdown("- Implement proactive outreach program with executive sponsorship")
                elif churn_rate > 0.25:
                    st.markdown("- 🟠 **High Priority**: Structured retention program needed")
                    st.markdown("- Implement satisfaction surveys and closed-loop feedback")
                    st.markdown("- Create re-engagement campaigns for at-risk users")
                    st.markdown("- Offer loyalty rewards or added-value services")
                    st.markdown("- Review pricing and contract terms for competitiveness")
                else:
                    st.markdown("- 🟡 **Moderate Priority**: Targeted improvements needed")
                    st.markdown("- Develop regular check-in program")
                    st.markdown("- Focus on product education and feature adoption")
                    st.markdown("- Implement usage-based early warning system")
                    st.markdown("- Create peer community and success resources")
        
        # Low-risk segments (bottom 2 by churn rate)
        low_risk_segments = segment_churn.tail(2).index.tolist()
        
        # For low-risk segments
        st.markdown("**Low-Risk Segments:**")
        for segment in low_risk_segments:
            churn_rate = segment_churn.loc[segment]
            segment_size = segment_counts.loc[segment]
            segment_pct = segment_size / segment_counts.sum()
            
            with st.expander(f"{segment} - Churn Rate: {churn_rate:.1%}, Size: {segment_size:,.0f} ({segment_pct:.1%} of customers)"):
                st.markdown("**Growth Opportunities:**")
                st.markdown("- 🟢 **Expansion Focus**: Target for upsell and cross-sell")
                st.markdown("- Develop advocacy and referral programs")
                st.markdown("- Create case studies and success stories")
                st.markdown("- Offer early access to new features and beta programs")
                st.markdown("- Maintain consistent engagement to preserve satisfaction")

if has_cohorts:
    # Cohort-based recommendations
    st.markdown("---")
    st.markdown("### 📆 Based on Cohort Analysis")
    
    # Get cohort data
    cohort_data = st.session_state.cohort_data
    cohort_column = st.session_state.cohort_column
    time_column = st.session_state.time_column
    
    if churn_cols:
        target_col = churn_cols[0]
        
        try:
            # Create a simplified cohort matrix (average across top 5 cohorts)
            top_cohorts = cohort_data[cohort_column].value_counts().head(5).index.tolist()
            cohort_subset = cohort_data[cohort_data[cohort_column].isin(top_cohorts)]
            
            # Create pivot table for top cohorts
            pivot = cohort_subset.pivot_table(
                values=target_col,
                index=cohort_column,
                columns=time_column,
                aggfunc='mean'
            )
            
            # Display pivot table
            st.write("Churn Rate by Cohort and Time Period (Top Cohorts):")
            st.dataframe(pivot.style.format("{:.1%}"))
            
            # Identify critical time periods
            avg_by_time = pivot.mean()
            critical_times = avg_by_time.sort_values(ascending=False).head(2).index.tolist()
            
            # Display critical time periods
            st.markdown("#### Critical Time Periods")
            st.markdown(f"The following time periods show the highest churn rates across cohorts:")
            
            for period in critical_times:
                churn_rate = avg_by_time.loc[period]
                st.markdown(f"- **{period}**: {churn_rate:.1%} average churn rate")
            
            # Calculate early vs. late churn pattern
            mid_point = len(avg_by_time) // 2
            early_periods = avg_by_time.index[:mid_point]
            late_periods = avg_by_time.index[mid_point:]
            
            early_churn = avg_by_time.loc[early_periods].mean()
            late_churn = avg_by_time.loc[late_periods].mean()
            
            # Display recommendations based on churn pattern
            st.markdown("#### Lifecycle-Based Recommendations")
            
            if early_churn > late_churn * 1.2:  # Early churn is at least 20% higher
                st.markdown("**Early Churn Pattern Detected**")
                st.markdown("Your data shows significantly higher churn in the early customer lifecycle, suggesting onboarding and initial value challenges.")
                
                with st.expander("Early Lifecycle Improvement Plan", expanded=True):
                    st.markdown("**Recommended Actions:**")
                    st.markdown("1. **Enhanced Onboarding Process**")
                    st.markdown("   - Develop a structured onboarding sequence with clear milestones")
                    st.markdown("   - Assign onboarding specialists for high-value accounts")
                    st.markdown("   - Create quick-win guides to demonstrate immediate value")
                    st.markdown("   - Implement 15, 30, and 60-day checkpoints")
                    
                    st.markdown("2. **Early Success Metrics**")
                    st.markdown("   - Define clear success criteria for the first 90 days")
                    st.markdown("   - Track feature adoption during initial period")
                    st.markdown("   - Establish early warning indicators for engagement issues")
                    st.markdown("   - Create customer health scores for new accounts")
                    
                    st.markdown("3. **Expectation Management**")
                    st.markdown("   - Align sales promises with actual onboarding timelines")
                    st.markdown("   - Set realistic timelines for value realization")
                    st.markdown("   - Provide clear product roadmaps to new customers")
                    st.markdown("   - Manage feature request expectations")
            
            elif late_churn > early_churn * 1.2:  # Late churn is at least 20% higher
                st.markdown("**Late Churn Pattern Detected**")
                st.markdown("Your data shows significantly higher churn in the later customer lifecycle, suggesting value degradation or competitive challenges.")
                
                with st.expander("Late Lifecycle Retention Plan", expanded=True):
                    st.markdown("**Recommended Actions:**")
                    st.markdown("1. **Value Reinforcement Program**")
                    st.markdown("   - Implement regular business reviews for mature accounts")
                    st.markdown("   - Create ROI calculators and value measurement tools")
                    st.markdown("   - Develop usage reports highlighting key benefits")
                    st.markdown("   - Create long-term success roadmaps with customers")
                    
                    st.markdown("2. **Maturity-Based Engagement**")
                    st.markdown("   - Offer advanced feature training for established users")
                    st.markdown("   - Create user communities and power-user programs")
                    st.markdown("   - Develop advanced use cases and resources")
                    st.markdown("   - Provide ongoing innovation and best practices")
                    
                    st.markdown("3. **Competitive Defense Strategy**")
                    st.markdown("   - Conduct regular competitive positioning reviews")
                    st.markdown("   - Create 'stay interviews' for long-term customers")
                    st.markdown("   - Develop preemptive renewal processes")
                    st.markdown("   - Offer long-term loyalty benefits and pricing")
            
            else:  # Relatively even churn pattern
                st.markdown("**Consistent Churn Pattern Detected**")
                st.markdown("Your data shows a relatively even churn pattern across the customer lifecycle.")
                
                with st.expander("Holistic Retention Strategy", expanded=True):
                    st.markdown("**Recommended Actions:**")
                    st.markdown("1. **Continuous Engagement Model**")
                    st.markdown("   - Implement a regular cadence of customer touchpoints")
                    st.markdown("   - Create usage-based engagement triggers")
                    st.markdown("   - Develop personalized content and resources")
                    st.markdown("   - Establish consistent communication channels")
                    
                    st.markdown("2. **Value Evolution Program**")
                    st.markdown("   - Create a progressive value roadmap for customers")
                    st.markdown("   - Align product development with evolving needs")
                    st.markdown("   - Develop regular feature refreshers and updates")
                    st.markdown("   - Provide ongoing benchmarking and best practices")
            
        except Exception as e:
            st.error(f"Error generating cohort-based recommendations: {str(e)}")

# Implementation plan section
st.markdown("---")
st.subheader("Implementation Plan")

# Create a simple implementation framework
st.markdown("""
### 90-Day Retention Improvement Plan

#### Immediate Actions (First 30 Days)
- Identify and reach out to the top 10% highest risk customers
- Implement basic churn risk scoring in your CRM or customer database
- Conduct exit interviews with recent churned customers
- Form a cross-functional retention task force

#### Short-Term Initiatives (30-60 Days)
- Develop segment-specific retention campaigns
- Review and optimize the onboarding process
- Implement regular customer health scoring
- Create dashboards to track retention metrics

#### Medium-Term Strategy (60-90 Days)
- Roll out enhanced customer success processes
- Implement systematic feedback collection
- Review pricing and packaging based on churn analysis
- Develop retention playbooks for the customer success team
""")

# ROI calculator
st.markdown("---")
st.subheader("Retention Program ROI Calculator")

# Simple ROI calculator
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Inputs")
    
    # Only show numeric inputs if we have valid numeric data
    if isinstance(total_customers, int) and isinstance(churn_rate, (int, float)):
        current_customers = st.number_input("Current customer count:", value=total_customers, min_value=1)
        current_churn_rate = st.number_input("Current annual churn rate (%):", value=churn_rate, min_value=0.0, max_value=100.0)
    else:
        current_customers = st.number_input("Current customer count:", value=1000, min_value=1)
        current_churn_rate = st.number_input("Current annual churn rate (%):", value=15.0, min_value=0.0, max_value=100.0)
    
    target_churn_reduction = st.slider("Target churn reduction (%):", 5, 50, 20)
    avg_customer_value = st.number_input("Average annual customer value ($):", value=1000, min_value=1)
    retention_program_cost = st.number_input("Annual retention program cost ($):", value=50000, min_value=0)

with col2:
    st.markdown("#### ROI Analysis")
    
    # Calculate ROI
    current_churn_decimal = current_churn_rate / 100
    reduced_churn_decimal = current_churn_decimal * (1 - target_churn_reduction / 100)
    
    # Customers saved annually
    customers_saved = current_customers * (current_churn_decimal - reduced_churn_decimal)
    
    # Revenue protected
    revenue_protected = customers_saved * avg_customer_value
    
    # Net benefit
    net_benefit = revenue_protected - retention_program_cost
    
    # ROI
    roi = (net_benefit / retention_program_cost) * 100 if retention_program_cost > 0 else float('inf')
    
    # Display results
    st.metric("Customers Saved Annually", f"{customers_saved:.0f}")
    st.metric("Revenue Protected Annually", f"${revenue_protected:,.0f}")
    st.metric("Net Benefit", f"${net_benefit:,.0f}")
    st.metric("ROI", f"{roi:.0f}%" if roi != float('inf') else "∞")

# Final call to action
st.markdown("---")
st.markdown("""
### Next Steps

1. **Share these insights** with your customer success and product teams
2. **Prioritize recommendations** based on your business goals and resources
3. **Implement a retention dashboard** to track improvements over time
4. **Establish a regular review process** to refine your retention strategy

Remember, even small improvements in retention can have a significant impact on your business growth and profitability.
""")
