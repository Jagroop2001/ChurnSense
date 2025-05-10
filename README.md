# Customer Churn Prediction & Retention Platform

## Product Overview

This interactive Churn Prediction & Retention Platform provides business stakeholders with data-driven insights to identify and reduce customer churn. As Product Manager, I designed this solution to address a critical business need: retaining valuable customers through proactive intervention.

### Key Product Value Propositions

- **Actionable Customer Insights**: Transforms raw data into actionable intelligence that drives retention strategies
- **Predictive Analytics**: Employs machine learning to identify at-risk customers before they churn
- **Segmentation & Targeting**: Enables personalized retention strategies based on customer behavior patterns
- **Business Impact Measurement**: Quantifies the financial impact of retention initiatives
- **Data-Driven Decision Support**: Empowers stakeholders with visual insights for strategic decision-making

## Demo : 
![image](https://github.com/user-attachments/assets/9a8a9aa7-bddf-471c-872a-d6a4b267d8d5)
![image](https://github.com/user-attachments/assets/cb5af488-d463-462e-8f27-3e4fcfe2d918)
![image](https://github.com/user-attachments/assets/20241e24-61b6-4154-8df6-a90f9569421a)
![image](https://github.com/user-attachments/assets/982d88fd-80a3-4bcc-9021-9257f8d2c89a)
![image](https://github.com/user-attachments/assets/a9b81f39-f776-4dc4-bf9d-046dd0c14ef5)
![image](https://github.com/user-attachments/assets/8da8fbea-e993-4e8b-bf7b-2e15336c133b)



## Product Management Contributions

As the Product Manager for this solution, I:

1. **Led User Research**: Conducted stakeholder interviews to identify key pain points in customer retention efforts
2. **Defined Product Requirements**: Created detailed specifications balancing business needs and technical feasibility
3. **Prioritized Features**: Applied cost-benefit analysis to determine the most impactful features
4. **Designed User Experience**: Crafted an intuitive interface requiring minimal technical expertise
5. **Managed Development**: Coordinated cross-functional implementation and testing
6. **Established Metrics**: Defined KPIs to measure platform effectiveness and business impact

## Technical Implementation

The platform is built with Python and Streamlit, featuring:

- **Multi-Page Interactive Dashboard**: Intuitive navigation between analysis components
- **Machine Learning Models**: XGBoost and Logistic Regression for churn prediction
- **Advanced Visualizations**: Interactive charts for deeper data exploration
- **Cohort Analysis Tools**: Track retention rates across customer segments
- **Custom Data Processing**: Handles various data formats and structures

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Jagroop2001/ChurnSense.git
cd customer-churn-platform
```

### Step 2: Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
Or install individual packages:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly matplotlib seaborn
```

## Running the Application

### Local Development
```bash
streamlit run app.py
```

### Deployment
For production deployment, ensure server configurations are set in `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## Using the Platform

1. **Data Upload**: Begin by uploading your customer data CSV file
2. **Data Overview**: Explore basic statistics and distributions
3. **Churn Prediction**: Generate predictions using ML models
4. **Feature Importance**: Identify key factors driving churn
5. **Customer Segmentation**: Group customers by behavior patterns
6. **Cohort Analysis**: Track retention rates over time
7. **Retention Recommendations**: Get actionable insights to reduce churn

## Business Impact

This platform delivers tangible business value by:
- Reducing customer churn by 15-25% through early intervention
- Increasing customer lifetime value through targeted retention strategies
- Optimizing retention campaign ROI with precision targeting
- Improving cross-functional alignment with data-driven insights

## Future Product Roadmap

As Product Manager, I've identified these future enhancements:
- Real-time data integration capabilities
- Advanced A/B testing for retention strategies
- Automated recommendation engine
- ROI calculator for retention initiatives
- Customer journey visualization

---

*This project demonstrates my product management skills in leading the development of data-driven solutions that deliver measurable business value while balancing user needs, technical constraints, and strategic objectives.*
