import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# --- PATH FIX: Ensures subfolder imports work ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your finalized functions
from freight_cost_prediction.inference.predict_freight import predict_freight
from invoice_flagging.inference.predict_invoice_risk import predict_invoice_risk

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("# 📦 Vendor Invoice Intelligence Portal")
st.markdown("## AI-Driven Freight Cost Prediction & Invoice Risk Flagging")

st.markdown("""
This internal analytics portal leverages machine learning to:
* **Forecast freight costs accurately**
* **Detect risky or abnormal vendor invoices**
* **Reduce financial leakage and manual workload**
""")
st.divider()

# --------------------------------------------------
# Sidebar & Model Selection
# --------------------------------------------------
st.sidebar.title("🔎 Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    ["Freight Cost Prediction", "Invoice Manual Approval Flag"]
)

st.sidebar.divider()
st.sidebar.markdown("""
### Business Impact
* 📉 **Improved cost forecasting**
* 🧾 **Reduced invoice fraud & anomalies**
* ⚙️ **Faster finance operations**
""")

# ==================================================
# FREIGHT COST PREDICTION
# ==================================================
if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")
    
    st.markdown("""
    **Objective:** Predict freight cost for a vendor invoice using **Invoice Dollars** to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        # Note: Added 'Quantity' just for UI consistency with screenshot, 
        # though your model only uses Dollars.
        col1, col2 = st.columns(2)
        with col1:
            qty = st.number_input("📦 Quantity", min_value=1, value=1200)
        with col2:
            dollars = st.number_input("💰 Invoice Dollars", min_value=1.0, value=18500.0)
            
        submit_freight = st.form_submit_button("🔮 Predict Freight Cost")

    if submit_freight:
        prediction = predict_freight(dollars)
        st.success("Prediction completed successfully.")
        st.metric(label="📊 Estimated Freight Cost", value=f"${prediction:,.2f}")

# ==================================================
# INVOICE RISK FLAGGING
# ==================================================
else:
    st.subheader("🚨 Invoice Manual Approval Prediction")
    
    st.markdown("""
    **Objective:** Predict whether a vendor invoice should be **flagged for manual approval** based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            invoice_qty = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight_cost = st.number_input("Freight Cost", min_value=0.0, value=1.73)
            
        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=352.95)
            total_item_qty = st.number_input("Total Item Quantity", min_value=1, value=162)
            
        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=2476.0)

        submit_flag = st.form_submit_button("🧠 Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
            'invoice_quantity': invoice_qty,
            'invoice_dollars': invoice_dollars,
            'Freight': freight_cost,
            'total_item_quantity': total_item_qty,
            'total_item_dollars': total_item_dollars
        }

        is_flagged, risk_score = predict_invoice_risk(input_data)

        if is_flagged == 1:
            st.error(f"🚨 FLAG FOR MANUAL APPROVAL (Risk Score: {risk_score:.2%})")
        else:
            st.success(f"✅ SAFE FOR AUTO-APPROVAL (Risk Score: {risk_score:.2%})")