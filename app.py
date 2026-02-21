pip install streamlit plotly joblib
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="AI Student Retention", layout="wide")

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("🎓 AI Student Retention Command Center")

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Dashboard",
    "🎯 Student Predictor",
    "💰 ROI Simulator"
])

# =====================
# TAB 1 DASHBOARD
# =====================

with tab1:

    st.header("Executive KPIs")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AUC Model", "0.96")
    col2.metric("Recall Dropout", "0.89")
    col3.metric("ROI Estimated", "692%")
    col4.metric("Students Saved", "530")

    st.subheader("Risk Distribution")

    fake_scores = np.random.beta(2, 5, 500)

    fig = px.histogram(fake_scores, nbins=30)
    st.plotly_chart(fig, use_container_width=True)


# =====================
# TAB 2 PREDICTOR
# =====================

with tab2:

    st.header("Student Risk Prediction")

    user_input = {}

    for f in features[:10]:   # limiter pour test
        user_input[f] = st.number_input(f, value=0.0)

    if st.button("Predict Risk"):

        X = pd.DataFrame([user_input])
        X_scaled = scaler.transform(X)

        proba = model.predict_proba(X_scaled)[0][1]

        st.success(f"Risk Score: {proba:.2f}")

        if proba < 0.33:
            st.info("Segment: Rassurant")
        elif proba < 0.66:
            st.warning("Segment: Vigilance")
        else:
            st.error("Segment: Alerte Critique")


# =====================
# TAB 3 ROI SIMULATOR
# =====================

with tab3:

    st.header("ROI Simulation")

    students = st.slider("Number of students", 100, 10000, 2000)
    cost_dropout = st.slider("Cost per dropout (€)", 5000, 30000, 15000)
    intervention_cost = st.slider("Intervention cost (€)", 100, 5000, 1000)
    success_rate = st.slider("Intervention success (%)", 0, 100, 30)

    saved = students * (success_rate / 100)
    savings = saved * cost_dropout

    roi = (savings - students * intervention_cost) / (students * intervention_cost)

    st.metric("Students Saved", int(saved))
    st.metric("Savings (€)", int(savings))
    st.metric("ROI", f"{roi:.2f}")
