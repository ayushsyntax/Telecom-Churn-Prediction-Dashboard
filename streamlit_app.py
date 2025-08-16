# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- Page Config ----------------
st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

# ---------------- Title ----------------
st.title("Telecom Churn Prediction")
st.write("A clean and simple interface to understand churn risk, model insights, and customer segments.")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Insights", "Customer Segments"])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.header("Prediction Form")
    st.caption("Fill in customer details to estimate churn probability.")

    with st.form("churn_form"):
        with st.expander("Personal Information", expanded=True):
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.checkbox("Senior Citizen")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with st.expander("Services"):
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No internet service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with st.expander("Billing & Contract"):
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 120.0, 50.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "PaperlessBilling": paperless_billing,
            "Partner": partner,
            "Dependents": dependents,
            "gender": gender,
            "SeniorCitizen": int(senior_citizen)
        }

        try:
            response = requests.post(API_URL, json=input_data)
            result = response.json()
            st.subheader("Prediction Result")
            st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            st.write(f"Risk Level: **{result['risk_level']}**")
            st.caption("Higher probability and high risk level suggest customer may need retention efforts.")
        except:
            st.error("Unable to connect to the prediction service.")

# ---------------- TAB 2: Model Insights ----------------
with tab2:
    st.header("Model Insights")
    st.write("The model uses CatBoost to predict customer churn.. SHAP values explain which features influence churn risk.")

    st.image("models/shap_summary.png")
    st.caption("Each dot is a customer. Left = lowers churn, Right = increases churn. Color shows feature value.")

    st.info(
        "Key interpretation:\n"
        "- Short contracts and high monthly charges increase churn risk.\n"
        "- Longer tenure reduces churn.\n"
        "- Fiber optic service users are at higher risk compared to DSL or no internet."
    )

# ---------------- TAB 3: Customer Segments ----------------
with tab3:
    st.header("Customer Segments")
    st.write("Customers are grouped into four clusters based on their service usage and spending patterns.")

    st.image("models/elbow_plot.png")
    st.caption("Elbow method shows 4 is the optimal number of customer clusters.")

    st.info(
        "Segment descriptions:\n"
        "- Segment 0: New customers on high-cost plans\n"
        "- Segment 1: Loyal low-spending customers\n"
        "- Segment 2: High-value but at-risk users\n"
        "- Segment 3: Premium loyalists"
    )
