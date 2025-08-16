# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model and preprocessor
model = joblib.load('models/champion_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

class CustomerInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str
    TechSupport: str
    PhoneService: str
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    StreamingTV: str
    StreamingMovies: str
    PaperlessBilling: str
    Partner: str
    Dependents: str
    gender: str
    SeniorCitizen: int

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    # Convert to DataFrame
    data = pd.DataFrame([customer.dict()])

    # Feature engineering 
    data['NumServices'] = data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies']].replace(
        {'Yes': 1, 'No': 0, 'No internet service': 0}).sum(axis=1)
    data['ContractNum'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    data['PaymentRisk'] = data['PaymentMethod'].map({
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    })
    data['HasFiberAndNoSupport'] = ((data['InternetService'] == 'Fiber optic') & 
                                    (data['TechSupport'] == 'No')).astype(int)
    data['tenure_group'] = pd.cut(data['tenure'], bins=[0, 12, 36, 72], labels=['New', 'Mid', 'Long']).astype(str)

    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        data[col] = data[col].map({'Yes': 1, 'No': 0})
    data['gender'] = data['gender'].map({'Female': 1, 'Male': 0})

    # Predict
    X = data.drop(['Churn'], axis=1, errors='ignore')
    churn_prob = model.predict_proba(X)[0][1]
    risk_level = "Critical" if churn_prob >= 0.75 else "High" if churn_prob >= 0.5 else "Medium" if churn_prob >= 0.25 else "Low"

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level
    }