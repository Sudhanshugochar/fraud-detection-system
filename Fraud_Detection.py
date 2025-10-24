import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Fraud Detection Prediction App")

st.markdown("Please enter the transaction detailes and use the predict button")
st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]) # Added more common transaction types for completeness
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0) # Completed the line based on typical app logic

oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

# The next part is usually where you put a button and prediction logic

if st.button("Predict Fraud"):
    # 1. Create a DataFrame from the inputs
    data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })
    
    # 2. Make the prediction
    # Assumes the loaded model handles all necessary preprocessing (e.g., one-hot encoding for 'type')
    prediction = model.predict(data)[0]
    prediction_proba = model.predict_proba(data)[0]

    # 3. Display the result
    if prediction == 1:
        st.error(f"**Prediction: FRAUDULENT TRANSACTION**")
        st.write(f"Probability of Fraud: {prediction_proba[1]*100:.2f}%")
    else:
        st.success(f"**Prediction: NOT FRAUDULENT**")
        st.write(f"Probability of Not Fraud: {prediction_proba[0]*100:.2f}%")

st.info("Note: This app requires a trained model saved as 'fraud_detection_pipeline.pkl' in the same directory.")