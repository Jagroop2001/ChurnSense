import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import shap, but continue if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not available. SHAP analysis features will be disabled.")

def plot_categorical_distribution(df, column, target_column=None):
    """
    Plot distribution of a categorical column
    
    Parameters:
    df (DataFrame): Input data
    column (str): Column to plot
    target_column (str): Target column for comparison (optional)
    
    Returns:
    fig: Plotly figure
    """
    if target_column and target_column in df.columns:
        # If target column is provided, create a grouped bar chart
        grouped_data = df.groupby([column, target_column]).size().reset_index(name='count')
        fig = px.bar(
            grouped_data, 
            x=column, 
            y='count', 
            color=target_column, 
            barmode='group',
            title=f'Distribution of {column} by {target_column}',
            labels={'count': 'Count', column: column.capitalize()},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    else:
        # Simple bar chart if no target column
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        fig = px.bar(
            value_counts, 
            x=column, 
            y='count',
            title=f'Distribution of {column}',
            labels={'count': 'Count', column: column.capitalize()},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    
    fig.update_layout(xaxis_title=column.capitalize(), yaxis_title='Count')
    return fig

def plot_numeric_distribution(df, column, target_column=None):
    """
    Plot distribution of a numeric column
    
    Parameters:
    df (DataFrame): Input data
    column (str): Column to plot
    target_column (str): Target column for comparison (optional)
    
    Returns:
    fig: Plotly figure
    """
    if target_column and target_column in df.columns:
        # If target column is provided, create histogram with color
        fig = px.histogram(
            df, 
            x=column, 
            color=target_column,
            marginal="box",
            title=f'Distribution of {column} by {target_column}',
            labels={column: column.capitalize()},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    else:
        # Simple histogram if no target column
        fig = px.histogram(
            df, 
            x=column,
            marginal="box",
            title=f'Distribution of {column}',
            labels={column: column.capitalize()},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    
    fig.update_layout(xaxis_title=column.capitalize(), yaxis_title='Count')
    return fig

def plot_correlation_heatmap(df, numeric_cols):
    """
    Plot correlation heatmap for numeric columns
    
    Parameters:
    df (DataFrame): Input data
    numeric_cols (list): List of numeric columns
    
    Returns:
    fig: Plotly figure
    """
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Correlation Heatmap'
    )
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    return fig

def plot_feature_importance(importance_df, top_n=15):
    """
    Plot feature importance
    
    Parameters:
    importance_df (DataFrame): DataFrame with feature importance
    top_n (int): Number of top features to show
    
    Returns:
    fig: Plotly figure
    """
    # Get top features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_features, 
        y='Feature', 
        x='Importance',
        orientation='h',
        title=f'Top {top_n} Feature Importance',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=500,
        width=800
    )
    
    return fig

def plot_shap_summary(shap_values, feature_names, max_display=20):
    """
    Plot SHAP summary plot for feature impact
    
    Parameters:
    shap_values: SHAP values from explainer
    feature_names (list): List of feature names
    max_display (int): Maximum number of features to display
    
    Returns:
    fig: Matplotlib figure or None if SHAP is not available
    """
    # Check if SHAP is available
    if not SHAP_AVAILABLE:
        # Create a fallback figure with a message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "SHAP library is not available.\nSHAP analysis features are disabled.", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                fontsize=14)
        ax.axis('off')
        return fig
        
    # If SHAP values are None (calculation failed), also show a message
    if shap_values is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "SHAP values calculation failed.\nCheck the logs for details.", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                fontsize=14)
        ax.axis('off')
        return fig
    
    try:
        plt.figure(figsize=(10, 8))
        
        # Create SHAP summary plot
        if isinstance(shap_values, shap.Explanation):
            # For new SHAP versions with Explanation object
            fig = plt.gcf()
            shap.summary_plot(shap_values, feature_names=feature_names, max_display=max_display, show=False)
        else:
            # For older SHAP versions
            fig = plt.gcf()
            shap.summary_plot(shap_values, feature_names=feature_names, max_display=max_display, show=False)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        # If an error occurs during plotting, return a figure with the error message
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Error creating SHAP plot: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                fontsize=14)
        ax.axis('off')
        return fig

def plot_confusion_matrix(conf_matrix, class_names=['Not Churned', 'Churned']):
    """
    Plot confusion matrix
    
    Parameters:
    conf_matrix (array): Confusion matrix from model evaluation
    class_names (list): Names of the classes
    
    Returns:
    fig: Plotly figure
    """
    # Create confusion matrix plot
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Viridis',
        title='Confusion Matrix'
    )
    
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=500
    )
    
    return fig

def plot_model_metrics_comparison(metrics_list):
    """
    Plot comparison of model metrics
    
    Parameters:
    metrics_list (list): List of dictionaries containing model metrics
    
    Returns:
    fig: Plotly figure
    """
    # Create a DataFrame for comparison
    compare_df = pd.DataFrame([
        {
            'Model': metrics['model_name'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'AUC': metrics['roc_auc']
        } for metrics in metrics_list
    ])
    
    # Melt the DataFrame for plotting
    melted_df = pd.melt(compare_df, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    # Create grouped bar chart
    fig = px.bar(
        melted_df,
        x='Metric',
        y='Value',
        color='Model',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Value': 'Score', 'Metric': 'Metric'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        height=500,
        width=800
    )
    
    return fig

def plot_customer_segments(df, segment_column, target_column=None):
    """
    Plot customer segments
    
    Parameters:
    df (DataFrame): Input data with segments
    segment_column (str): Column containing segment labels
    target_column (str): Target column for coloring (optional)
    
    Returns:
    fig: Plotly figure
    """
    # Count distribution by segment
    segment_counts = df[segment_column].value_counts().reset_index()
    segment_counts.columns = [segment_column, 'Count']
    
    # If target column is provided, calculate churn rate by segment
    if target_column and target_column in df.columns:
        segment_churn = df.groupby(segment_column)[target_column].mean().reset_index()
        segment_churn.columns = [segment_column, 'Churn Rate']
        
        # Create a subplot with two charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Segment Distribution', 'Churn Rate by Segment'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=segment_counts[segment_column],
                y=segment_counts['Count'],
                name='Count',
                marker_color='royalblue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=segment_churn[segment_column],
                y=segment_churn['Churn Rate'],
                name='Churn Rate',
                marker_color='indianred'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Customer Segment Analysis',
            height=500,
            width=1000
        )
        
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Churn Rate', range=[0, 1], row=1, col=2)
    
    else:
        # Simple pie chart if no target column
        fig = px.pie(
            segment_counts,
            values='Count',
            names=segment_column,
            title='Customer Segment Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=500,
            width=700
        )
    
    return fig

def plot_cohort_analysis(df, cohort_column, time_column, target_column):
    """
    Plot cohort analysis of churn rates
    
    Parameters:
    df (DataFrame): Input data
    cohort_column (str): Column to define cohorts (e.g., signup_month)
    time_column (str): Column for time progression (e.g., tenure)
    target_column (str): Target column (churn)
    
    Returns:
    fig: Plotly figure
    """
    # Aggregate data: for each cohort and time period, calculate churn rate
    cohort_data = df.groupby([cohort_column, time_column])[target_column].mean().reset_index()
    
    # Pivot the data for heatmap visualization
    pivot_data = cohort_data.pivot(index=cohort_column, columns=time_column, values=target_column)
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x=time_column, y=cohort_column, color="Churn Rate"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn_r',  # Red for high churn, green for low churn
        title=f'Cohort Analysis: Churn Rate by {cohort_column} and {time_column}'
    )
    
    fig.update_layout(
        xaxis_title=time_column.capitalize(),
        yaxis_title=cohort_column.capitalize(),
        height=600,
        width=800
    )
    
    return fig

def plot_churn_prediction_distribution(probabilities, threshold=0.5):
    """
    Plot distribution of churn probabilities
    
    Parameters:
    probabilities (array): Predicted churn probabilities
    threshold (float): Decision threshold for classification
    
    Returns:
    fig: Plotly figure
    """
    # Create a DataFrame with probabilities
    pred_df = pd.DataFrame({
        'Churn Probability': probabilities,
        'Prediction': ['Churned' if p >= threshold else 'Not Churned' for p in probabilities]
    })
    
    # Create histogram
    fig = px.histogram(
        pred_df,
        x='Churn Probability',
        color='Prediction',
        marginal='box',
        title='Distribution of Churn Probabilities',
        labels={'Churn Probability': 'Probability of Churn'},
        color_discrete_map={
            'Churned': 'indianred',
            'Not Churned': 'royalblue'
        }
    )
    
    # Add vertical line for threshold
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='black',
        annotation_text=f'Threshold: {threshold}',
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        height=500,
        width=800
    )
    
    return fig
