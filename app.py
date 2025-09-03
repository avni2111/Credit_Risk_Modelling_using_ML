import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("XGBoost_credit_model.pkl")
encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# App title
st.title("Credit Risk Prediction App")
st.write("Enter application information to predict if the credit risk is GOOD or BAD")

# User inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", encoders["Sex"].classes_.tolist()) 
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", encoders["Housing"].classes_.tolist())
saving_accounts = st.selectbox("Saving Accounts", encoders["Saving accounts"].classes_.tolist())
checking_account = st.selectbox("Checking Accounts", encoders["Checking account"].classes_.tolist())
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# Transform inputs
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# Prediction
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("✅ The predicted credit risk is: **GOOD (Lower Risk)**")
    else:
        st.error("⚠️ The predicted credit risk is: **BAD (Higher Risk)**")
