import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualizations import plot_cohort_analysis

st.set_page_config(page_title="Cohort Analysis", page_icon="📆", layout="wide")

st.title("Cohort Analysis")

st.markdown("""
This section allows you to analyze churn patterns across different customer cohorts.
Cohort analysis helps identify how churn varies based on when customers joined and their lifecycle stage.
""")

# Check if data is available
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload your data in the Data Upload section first.")
    st.stop()

# Identify potential cohort and time columns
data = st.session_state.data
all_columns = data.columns.tolist()

# Try to identify the target column (churn indicator)
churn_cols = [col for col in all_columns if 'churn' in col.lower()]
target_col = churn_cols[0] if churn_cols else None

# Try to identify potential cohort columns (date, month, year, etc.)
cohort_candidates = [col for col in all_columns if any(term in col.lower() for term in 
                                                     ['date', 'month', 'year', 'join', 'start', 'signup', 'register', 'onboard'])]

# Try to identify potential time/tenure columns
time_candidates = [col for col in all_columns if any(term in col.lower() for term in 
                                                   ['tenure', 'time', 'period', 'duration', 'month', 'day', 'year'])]

# Cohort selection section
st.subheader("Define Cohorts")

# Option 1: Use existing cohort column if available
if cohort_candidates:
    st.write("Option 1: Use an existing column that defines when customers joined")
    existing_cohort_col = st.selectbox(
        "Select a column that indicates when customers joined:",
        cohort_candidates + ['None of these']
    )
    
    use_existing_cohort = existing_cohort_col != 'None of these'
else:
    use_existing_cohort = False
    existing_cohort_col = None

# Option 2: Create cohorts from a date column
date_cols = [col for col in all_columns if data[col].dtype == 'datetime64[ns]' or 
             ('date' in col.lower() or 'time' in col.lower() or 'day' in col.lower())]

if date_cols:
    st.write("Option 2: Create cohorts from a date column")
    date_column = st.selectbox(
        "Select a date column to create cohorts:",
        date_cols + ['None of these']
    )
    
    use_date_column = date_column != 'None of these'
else:
    use_date_column = False
    date_column = None

# Option 3: Create custom cohorts
st.write("Option 3: Create custom cohorts based on a feature")
custom_cohort_col = st.selectbox(
    "Select a column to create custom cohorts:",
    [col for col in all_columns if col not in (churn_cols or [])] + ['None of these']
)

use_custom_cohort = custom_cohort_col != 'None of these'

# Select time/tenure column
st.subheader("Select Progression Dimension")
st.write("Choose a column that shows customer progression over time (e.g., tenure)")

if time_candidates:
    time_column = st.selectbox(
        "Select a time/tenure column:",
        time_candidates + [col for col in all_columns if col not in time_candidates],
        index=0 if time_candidates else 0
    )
else:
    time_column = st.selectbox(
        "Select a time/tenure column:",
        [col for col in all_columns if col not in (churn_cols or [])]
    )

# Button to generate cohort analysis
if st.button("Generate Cohort Analysis"):
    # Check if we have a target column
    if not target_col:
        st.error("No churn column detected. Please ensure your dataset has a column indicating churn.")
        st.stop()
    
    # Prepare data for cohort analysis
    cohort_data = data.copy()
    
    # Define cohort column based on selection
    if use_existing_cohort:
        cohort_column = existing_cohort_col
        st.write(f"Using existing cohort column: {cohort_column}")
    elif use_date_column:
        # Try to convert column to datetime if it's not already
        try:
            if cohort_data[date_column].dtype != 'datetime64[ns]':
                cohort_data[date_column] = pd.to_datetime(cohort_data[date_column])
            
            # Extract year-month to create cohorts
            cohort_data['Cohort'] = cohort_data[date_column].dt.to_period('M').astype(str)
            cohort_column = 'Cohort'
            st.write(f"Created monthly cohorts from date column: {date_column}")
        except Exception as e:
            st.error(f"Error creating date-based cohorts: {str(e)}")
            st.stop()
    elif use_custom_cohort:
        # Use values from the selected column to create cohorts
        cohort_column = custom_cohort_col
        st.write(f"Using custom cohort column: {cohort_column}")
    else:
        st.error("Please select a valid option for creating cohorts.")
        st.stop()
    
    # Make sure the time column is numeric
    if not pd.api.types.is_numeric_dtype(cohort_data[time_column]):
        st.error(f"Time column '{time_column}' must be numeric. Please select a different column.")
        st.stop()
    
    # Discretize time column if it has too many unique values
    time_unique_values = cohort_data[time_column].nunique()
    if time_unique_values > 10:
        # Bin the time values into categories
        max_time = cohort_data[time_column].max()
        bin_width = max(1, max_time // 10)  # Ensure at least 1
        
        # Create bins and labels
        bins = list(range(0, int(max_time) + bin_width + 1, bin_width))
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
        
        # Create the binned column
        cohort_data['Time_Binned'] = pd.cut(
            cohort_data[time_column], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        original_time_column = time_column
        time_column = 'Time_Binned'
        st.info(f"Time column '{original_time_column}' has been binned into {len(labels)} categories.")
    
    # Make sure cohort column doesn't have too many unique values
    cohort_unique_values = cohort_data[cohort_column].nunique()
    if cohort_unique_values > 10:
        # Keep only the top cohorts
        top_cohorts = cohort_data[cohort_column].value_counts().head(10).index.tolist()
        cohort_data = cohort_data[cohort_data[cohort_column].isin(top_cohorts)]
        st.info(f"Limited analysis to top 10 cohorts as there were {cohort_unique_values} unique cohorts.")
    
    # Store the prepared data
    st.session_state.cohort_data = cohort_data
    st.session_state.cohort_column = cohort_column
    st.session_state.time_column = time_column
    
    st.success("Cohort analysis data prepared!")

# Display cohort analysis if data is ready
if 'cohort_data' in st.session_state and st.session_state.cohort_data is not None:
    st.markdown("---")
    st.subheader("Cohort Analysis Results")
    
    cohort_data = st.session_state.cohort_data
    cohort_column = st.session_state.cohort_column
    time_column = st.session_state.time_column
    
    # Create cohort churn analysis
    # Pivot table: cohorts as rows, time periods as columns, churn rate as values
    try:
        # Convert possible categorical time column to string for proper ordering
        if isinstance(cohort_data[time_column].dtype, pd.CategoricalDtype):
            cohort_data[time_column] = cohort_data[time_column].astype(str)
            
        # Create pivot table
        cohort_pivot = cohort_data.pivot_table(
            values=target_col,
            index=cohort_column,
            columns=time_column,
            aggfunc='mean'  # Mean of churn column gives churn rate
        )
        
        # Display the pivot table
        st.write("Churn Rate by Cohort and Time Period:")
        st.dataframe(cohort_pivot.style.format("{:.2%}"))
        
        # Create heatmap visualization
        fig = px.imshow(
            cohort_pivot,
            labels=dict(x=time_column, y=cohort_column, color="Churn Rate"),
            x=cohort_pivot.columns,
            y=cohort_pivot.index,
            color_continuous_scale='RdYlGn_r',  # Red for high churn, green for low churn
            title=f'Cohort Analysis: Churn Rate by {cohort_column} and {time_column}'
        )
        
        fig.update_layout(
            xaxis_title=time_column,
            yaxis_title=cohort_column,
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional cohort analysis: Retention rate
        st.subheader("Retention Analysis by Cohort")
        
        # Calculate retention rate (1 - churn rate)
        retention_pivot = 1 - cohort_pivot
        
        # Display the retention pivot table
        st.write("Retention Rate by Cohort and Time Period:")
        st.dataframe(retention_pivot.style.format("{:.2%}"))
        
        # Retention rate heatmap
        fig = px.imshow(
            retention_pivot,
            labels=dict(x=time_column, y=cohort_column, color="Retention Rate"),
            x=retention_pivot.columns,
            y=retention_pivot.index,
            color_continuous_scale='RdYlGn',  # Green for high retention, red for low retention
            title=f'Cohort Analysis: Retention Rate by {cohort_column} and {time_column}'
        )
        
        fig.update_layout(
            xaxis_title=time_column,
            yaxis_title=cohort_column,
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Line chart for cohort churn rates over time
        st.subheader("Cohort Churn Trend Analysis")
        
        # Line chart with one line per cohort
        fig = px.line(
            cohort_pivot.reset_index().melt(
                id_vars=cohort_column, 
                value_vars=cohort_pivot.columns,
                var_name=time_column,
                value_name='Churn Rate'
            ),
            x=time_column,
            y='Churn Rate',
            color=cohort_column,
            markers=True,
            title=f'Churn Rate Trends by Cohort Over {time_column}'
        )
        
        fig.update_layout(
            xaxis_title=time_column,
            yaxis_title='Churn Rate',
            height=500,
            width=800,
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights from cohort analysis
        st.subheader("Cohort Analysis Insights")
        
        # Calculate average churn rate by cohort and time period
        avg_churn_by_cohort = cohort_pivot.mean(axis=1).sort_values(ascending=False)
        avg_churn_by_time = cohort_pivot.mean(axis=0).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Average Churn Rate by Cohort:")
            fig = px.bar(
                x=avg_churn_by_cohort.index,
                y=avg_churn_by_cohort.values,
                labels={'x': cohort_column, 'y': 'Avg. Churn Rate'},
                title=f'Average Churn Rate by {cohort_column}',
                color=avg_churn_by_cohort.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("Average Churn Rate by Time Period:")
            fig = px.bar(
                x=avg_churn_by_time.index,
                y=avg_churn_by_time.values,
                labels={'x': time_column, 'y': 'Avg. Churn Rate'},
                title=f'Average Churn Rate by {time_column}',
                color=avg_churn_by_time.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Provide key insights
        st.write("Key Insights:")
        
        # Identify highest churn cohort
        highest_churn_cohort = avg_churn_by_cohort.index[0]
        highest_churn_rate = avg_churn_by_cohort.iloc[0]
        st.markdown(f"- **Highest Risk Cohort**: {highest_churn_cohort} with average churn rate of {highest_churn_rate:.2%}")
        
        # Identify lowest churn cohort
        lowest_churn_cohort = avg_churn_by_cohort.index[-1]
        lowest_churn_rate = avg_churn_by_cohort.iloc[-1]
        st.markdown(f"- **Lowest Risk Cohort**: {lowest_churn_cohort} with average churn rate of {lowest_churn_rate:.2%}")
        
        # Identify critical time periods
        highest_churn_time = avg_churn_by_time.index[0]
        highest_time_churn_rate = avg_churn_by_time.iloc[0]
        st.markdown(f"- **Critical Time Period**: {highest_churn_time} with average churn rate of {highest_time_churn_rate:.2%}")
        
        # Look for early vs. late churn patterns
        early_periods = avg_churn_by_time.index[:len(avg_churn_by_time)//2]
        late_periods = avg_churn_by_time.index[len(avg_churn_by_time)//2:]
        
        early_churn = cohort_pivot[early_periods].mean().mean()
        late_churn = cohort_pivot[late_periods].mean().mean()
        
        if early_churn > late_churn:
            st.markdown(f"- **Early Churn Pattern**: Customers are {early_churn/late_churn:.1f}x more likely to churn in early periods compared to later periods, suggesting onboarding issues")
        else:
            st.markdown(f"- **Late Churn Pattern**: Customers are {late_churn/early_churn:.1f}x more likely to churn in later periods compared to early periods, suggesting value degradation over time")
        
    except Exception as e:
        st.error(f"Error generating cohort analysis: {str(e)}")

# Show next steps
st.markdown("---")
st.markdown("""
### Next Steps

Now that you've analyzed churn patterns across different cohorts, you can:

1. View **Retention Recommendations** based on your cohort analysis
2. Develop targeted strategies for high-risk cohorts
3. Address specific time periods where churn spikes occur
""")
