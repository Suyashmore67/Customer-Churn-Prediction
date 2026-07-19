import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved model, scaler, threshold
# -----------------------------
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

with open("model/threshold.txt", "r") as f:
    threshold = float(f.read())

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their details.")

# -----------------------------
# User Input Form
# -----------------------------
st.subheader("Enter Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

# -----------------------------
# Convert inputs to DataFrame
# (Must match training columns)
# -----------------------------

# Create empty input with all features
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

# Fill numeric features
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# Contract
if contract == "One year":
    input_data["Contract_One year"] = 1
elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Payment Method
input_data[f"PaymentMethod_{payment_method}"] = 1

# Internet Service
if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_data["InternetService_No"] = 1


# -----------------------------
# Scale numerical features
# -----------------------------
input_data_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    churn_prob = model.predict_proba(input_data_scaled)[0][1]

    st.write(f"### 🔍 Churn Probability: **{churn_prob:.2%}**")

    if churn_prob >= threshold:
        st.error("⚠️ High Risk: Customer is likely to churn")
    else:
        st.success("✅ Low Risk: Customer is likely to stay")
