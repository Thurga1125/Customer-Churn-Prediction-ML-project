# Customer Churn Prediction

A machine learning project to predict customer churn in the telecommunications industry using supervised classification algorithms.

## Project Overview

Customer churn represents the phenomenon where customers discontinue their subscription services, leading to significant revenue loss for telecom companies. This project develops predictive models to identify customers at high risk of churning by analyzing customer demographics, service subscriptions, account information, and billing patterns.

## Dataset

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/mosapabdelghany/telcom-customerchurn-dataset)
- **Size**: 7,043 customers, 21 columns (19 features + customerID + target)
- **Original Features**: 19
- **Target Variable**: Churn (Yes/No)

### Features

| Category | Features |
|----------|----------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Account | Contract, PaperlessBilling, PaymentMethod, tenure |
| Billing | MonthlyCharges, TotalCharges |

## Project Structure

```
Customer-Churn-Prediction-ML-project/
├── data/
│   └── Telco_Cusomer_Churn.csv
├── notebooks/
│   └── churn_prediction.ipynb
├── models/
│   ├── logistic_regression_churn_model.pkl
│   ├── random_forest_churn_model.pkl
│   └── scaler.pkl
├── results/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_internet.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── cross_validation.png
│   ├── error_analysis.png
│   └── model_comparison.png
├── venv/
├── README.md
└── requirements.txt
```

## Algorithms Implemented

### 1. Logistic Regression
- Linear classification algorithm
- Best performing model in this project
- Provides interpretable coefficients

### 2. Random Forest
- Ensemble learning method
- Provides feature importance rankings
- Robust to outliers

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **80.55%** | **65.72%** | **55.88%** | **60.40%** | **0.842** |
| Random Forest | 78.28% | 61.33% | 49.20% | 54.60% | 0.826 |

**Best Model**: Logistic Regression with 80.55% accuracy and 0.842 ROC AUC

## Key Findings

1. **Churn Rate**: 26.54% of customers churned
2. **Contract Type**: Month-to-month contracts have the highest churn rate
3. **Tenure**: New customers (low tenure) are more likely to churn
4. **Internet Service**: Fiber optic customers show elevated churn rates
5. **Monthly Charges**: Higher charges correlate with increased churn probability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Thurga1125/Customer-Churn-Prediction-ML-project
cd Customer-Churn-Prediction-ML-project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

2. Run all cells to:
   - Load and explore the dataset
   - Preprocess the data
   - Train both models
   - Evaluate and compare results
   - Generate visualizations

3. Use saved models for predictions:
```python
import joblib

# Load model and scaler
model = joblib.load('models/logistic_regression_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Make predictions
predictions = model.predict(X_new)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Business Recommendations

1. **Target Month-to-Month Customers**: Offer incentives to convert to longer-term contracts
2. **Focus on New Customers**: Implement onboarding programs and early engagement
3. **Review Fiber Optic Service**: Investigate service quality and pricing
4. **Monitor High Charges**: Review pricing strategy for high-paying customers
5. **Proactive Retention**: Use model predictions to identify and retain at-risk customers

## Authors

- Uduwawala W.B.W.M.R.S.H (EG/2022/5377)
- Thurga R. (EG/2022/5374)

## Course

EC5203 Machine Learning Project - Group 38

## License

This project is for educational purposes as part of the EC5203 Machine Learning course.
