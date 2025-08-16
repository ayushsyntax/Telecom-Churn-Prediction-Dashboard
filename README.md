# Telecom Churn Prediction Project

## Overview

This project analyzes customer data from a telecommunications company to predict customer churn using machine learning. The goal is to identify high-risk customers and enable targeted retention strategies.

The workflow includes:
- Data exploration and preprocessing
- Feature engineering
- Model training and evaluation
- Customer segmentation
- Model interpretation with SHAP
- Deployment-ready model persistence

---

## Dataset

### Source
Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Key Features
| Feature | Description |
|---------|-------------|
| `customerID` | Unique identifier |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographic information |
| `tenure` | Months with the company |
| `PhoneService`, `InternetService`, `OnlineSecurity`, etc. | Service usage |
| `Contract`, `PaperlessBilling`, `PaymentMethod` | Billing details |
| `MonthlyCharges`, `TotalCharges` | Financial metrics |
| `Churn` | Target variable (Yes/No) |

### Dataset Statistics
- **Rows**: 7,032 customers
- **Columns**: 21 features
- **Churn distribution**: 5163 (No), 1869 (Yes)
- **Class imbalance**: Approximately 26% churn rate

---

## Methodology

### Exploratory Data Analysis (EDA)

#### Initial Observations
- **Tenure distribution**: Median tenure is 29 months
- **Monthly charges**: Range from $18.25 to $118.75
- **Total charges**: Skewed distribution with many low values

#### Key Insights
- Customers with month-to-month contracts have higher churn rates
- Fiber optic internet users show higher churn when lacking support services
- Electronic check payment method correlates with significantly higher churn (45.3%)
- Customers without online security have 41.8% churn rate vs. 14.6% with security

#### Visualizations
- Churn distribution by categorical variables
- Tenure vs. monthly charges scatter plot
- Correlation matrix of numerical features
- Heatmaps showing churn rates by contract type and payment method

---

### Feature Engineering

#### Derived Features
- **NumServices**: Count of services used (0-6)
- **tenure_group**: Categorized as New (<1yr), Mid (1-3yrs), Long (>3yrs)
- **ContractNum**: 0=Month-to-month, 1=One year, 2=Two year
- **PaymentRisk**: Risk score for payment methods (Electronic check = highest risk)
- **HasFiberAndNoSupport**: Flag for high-risk combination of fiber optic + no tech support

#### Preprocessing Pipeline
- **Numerical features**: StandardScaler normalization
- **Categorical features**: OneHotEncoder with drop='first' to avoid multicollinearity
- **Missing values**: TotalCharges filled with 0 after conversion to numeric

---

### Model Development

#### Evaluation Strategy
- **Cross-validation**: Stratified K-Fold (5 splits)
- **Metrics**: ROC AUC, Precision, Recall, F1-score
- **Training split**: 80% train, 20% test (stratified)
- **Random state**: 42 for reproducibility

#### Models Compared
| Model | Test AUC | Recall (Churn) | Precision (Churn) | F1 (Churn) |
|-------|----------|----------------|-------------------|------------|
| Logistic Regression | 0.834 | 50.8% | 65.3% | 57.1% |
| Random Forest | 0.821 | 46.3% | 67.3% | 54.8% |
| XGBoost | 0.807 | 67.1% | 50.7% | 57.8% |
| LightGBM | 0.828 | 74.1% | 50.5% | 60.1% |
| **CatBoost** | **0.830** | **77.0%** | **51.5%** | **61.7%** |

#### Champion Model Selection
- **Selected**: CatBoost Classifier
- **Rationale**: Highest recall for churn class (77.0%), critical for identifying at-risk customers
- **Configuration**:
  - `scale_pos_weight`: 2.75 (accounts for class imbalance)
  - `random_state`: 42
  - `verbose`: 0 (suppresses output during training)

---

### Model Performance

#### Final Results
- **Test AUC**: 0.830
- **Accuracy**: 75%
- **Classification Report**:
  ```
  precision    recall  f1-score   support
           0       0.90      0.74      0.81      1033
           1       0.52      0.77      0.62       374
  accuracy                           0.75      1407
  macro avg       0.71      0.75      0.71      1407
  weighted avg       0.80      0.75      0.76      1407
  ```

#### Interpretation
- **Recall of 77%**: Model successfully identifies 77% of actual churners
- **Precision of 52%**: Half of predicted churners actually churn
- **F1-score of 62%**: Balanced measure of precision and recall
- **AUC of 0.83**: Strong discriminatory power between churn and non-churn customers

---

### Customer Segmentation

#### Clustering Approach
- **Algorithm**: K-Means clustering
- **Features**: tenure, MonthlyCharges, NumServices, ContractNum, PaymentRisk, SeniorCitizen, Partner
- **Optimal clusters**: k=4 determined via elbow method

#### Segment Profiles
| Segment | Characteristics | Churn Risk |
|---------|------------------|------------|
| **0** | Long-term, high spend, multiple services | Low |
| **1** | Low spend, few services, stable contracts | Medium |
| **2** | Seniors, medium spend, limited services | High |
| **3** | New customers, high churn risk | Critical |

---

### Model Interpretability

#### SHAP Analysis
- **Purpose**: Understand feature contributions to predictions
- **Method**: SHAP values on a sample of 100 customers
- **Key Findings**:
  - Negative impact: tenure, contract length, payment method risk
  - Positive impact: fiber optic service, lack of support services
  - Most influential features: tenure, contract type, payment method

---

## Implementation

### File Structure
```
TelecomChurnPrediction/
├── models/
│   ├── champion_model.pkl
│   ├── preprocessor.pkl
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   ├── elbow_plot.png
│   └── shap_summary.png
├── app.py
├── streamlit_app.py
├── requirements.txt
├── Telco_Customer_Churn.ipynb
└── README.md
```

### Dependencies
```txt
pandas
numpy
scikit-learn
catboost
seaborn
matplotlib
shap
joblib
streamlit
fastapi
uvicorn
```

### Model Persistence
- **Format**: joblib pickle files
- **Files saved**:
  - `champion_model.pkl`: Trained CatBoost model
  - `preprocessor.pkl`: Feature preprocessing pipeline
  - `kmeans.pkl`: Customer segmentation model
  - `scaler.pkl`: StandardScaler for clustering
- **Version compatibility**: Models trained with scikit-learn 1.6.1

---

