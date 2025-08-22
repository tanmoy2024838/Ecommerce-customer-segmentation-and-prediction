# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:50:42 2025

@author: TANMOY
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# --- Title and Description ---
st.title("🛍️ E-commerce Customer Analytics Dashboard")
st.write("This application provides a dashboard for customer segmentation and predictive analytics.")

st.markdown("---")

# --- Load Data and Models ---
@st.cache_data
def load_data_and_models():
    """Load all necessary data and trained models."""
    try:
        # Load datasets
        df_rfm = pd.read_csv('rfm_segmentation.csv')
        df_original = pd.read_csv('final_cleaned_data.csv')
        df_original['InvoiceDate'] = pd.to_datetime(df_original['InvoiceDate'])

        # Load models and scalers
        model_clv = joblib.load('clv_model.pkl')
        scaler_clv = joblib.load('scaler_clv.pkl')
        model_churn = joblib.load('churn_model.pkl')
        scaler_churn = joblib.load('scaler.pkl')
        model_next_purchase = joblib.load('next_purchase_model.pkl')
        scaler_next_purchase = joblib.load('scaler_next_purchase.pkl')

        return df_rfm, df_original, model_clv, scaler_clv, model_churn, scaler_churn, model_next_purchase, scaler_next_purchase
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure all data and model files are in the same directory.")
        return None, None, None, None, None, None, None, None

df_rfm, df_original, model_clv, scaler_clv, model_churn, scaler_churn, model_next_purchase, scaler_next_purchase = load_data_and_models()

# --- Application Logic ---
if df_rfm is not None:
    # --- User Selection ---
    st.sidebar.header("Customer Information")
    customer_ids = sorted(df_rfm['CustomerID'].unique().tolist())
    selected_customer_id = st.sidebar.selectbox("Select a Customer ID:", customer_ids)

    # --- Find and Display Customer Data ---
    customer_data = df_rfm[df_rfm['CustomerID'] == selected_customer_id].iloc[0]

    st.header("👤 Customer Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recency (Days)", f"{customer_data['Recency']:.0f}")
    with col2:
        st.metric("Frequency", f"{customer_data['Frequency']:.0f}")
    with col3:
        st.metric("Monetary ($)", f"{customer_data['Monetary']:.2f}")

    st.write(f"**Customer Segment:** {customer_data['CustomerSegment']}")

    st.markdown("---")

    # --- Make Predictions ---
    st.header(" Predictive Analytics")
    if st.button("Generate Predictions"):
        with st.spinner('Calculating predictions...'):
            # Prepare the input for the models
            input_data = pd.DataFrame({
                'Recency': [customer_data['Recency']],
                'Frequency': [customer_data['Frequency']],
                'Monetary': [customer_data['Monetary']]
            })

            # Predict Customer Lifetime Value (CLV)
            scaled_clv = scaler_clv.transform(input_data)
            predicted_clv = model_clv.predict(scaled_clv)[0]

            # Predict Churn Risk
            scaled_churn = scaler_churn.transform(input_data)
            churn_risk_prob = model_churn.predict_proba(scaled_churn)[:, 1][0]
            churn_risk_percent = churn_risk_prob * 100

            # Predict Next Purchase Date
            scaled_next_purchase = scaler_next_purchase.transform(input_data)
            predicted_days = model_next_purchase.predict(scaled_next_purchase)[0]
            
            latest_purchase_date = df_original[df_original['CustomerID'] == selected_customer_id]['InvoiceDate'].max()
            if not pd.isna(latest_purchase_date):
                next_purchase_date = latest_purchase_date + timedelta(days=max(0, predicted_days))
            else:
                next_purchase_date = "N/A"
            

            # Display predictions
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            with col_pred1:
                st.metric("Customer Lifetime Value", f"${predicted_clv:.2f}")
            with col_pred2:
                st.metric("Churn Risk", f"{churn_risk_percent:.2f}%")
            with col_pred3:
                st.metric("Predicted Next Purchase", next_purchase_date.strftime('%Y-%m-%d') if isinstance(next_purchase_date, datetime) else next_purchase_date)

            st.success("Predictions generated successfully!")

    st.markdown("---")

    st.info("**How to use:** Select a customer ID from the sidebar and click 'Generate Predictions' to see their forecasted values.")
else:
    st.warning("Please check your file paths and ensure all required files exist.")