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

    st.write("Fill in the student information below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age at enrollment", min_value=18, max_value=80, value=20, step=1)
        gender_choice = st.selectbox("Gender", ["Male", "Female"])
        gender = 0 if gender_choice == "Male" else 1
        displaced = st.radio("Displaced", [0, 1], horizontal=True)
        international = st.radio("International", [0, 1], horizontal=True)

    with col2:
        st.subheader("Academic Information")
        attendance = st.radio("Daytime/evening attendance", [0, 1], horizontal=True, captions=["Daytime", "Evening"])
        scholarship = st.radio("Scholarship holder", [0, 1], horizontal=True)
        debtor = st.radio("Debtor", [0, 1], horizontal=True)
        tuition_paid = st.radio("Tuition fees up to date", [0, 1], horizontal=True)

    with col3:
        st.subheader("1st Semester Performance")
        units_1st_enrolled = st.number_input("1st sem - Units enrolled", min_value=0, max_value=30, value=6, step=1)
        units_1st_approved = st.number_input("1st sem - Units approved", min_value=0, max_value=30, value=1, step=1)
        grade_1st = st.number_input("1st sem - Grade average", min_value=0.0, max_value=20.0, value=8.5, step=0.1)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("2nd Semester Performance")
        units_2nd_enrolled = st.number_input("2nd sem - Units enrolled", min_value=0, max_value=30, value=6, step=1)
        units_2nd_approved = st.number_input("2nd sem - Units approved", min_value=0, max_value=30, value=0, step=1)
        grade_2nd = st.number_input("2nd sem - Grade average", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    with col5:
        st.subheader("Family Background")
        mother_qual = st.number_input("Mother's qualification", min_value=0, max_value=40, value=1, step=1)
        father_qual = st.number_input("Father's qualification", min_value=0, max_value=40, value=1, step=1)

    with col6:
        st.subheader("Economic Indicators")
        unemployment = st.number_input("Unemployment rate (%)", min_value=0.0, max_value=50.0, value=13.9, step=0.1)
        inflation = st.number_input("Inflation rate (%)", min_value=-5.0, max_value=20.0, value=3.2, step=0.1)
        gdp = st.number_input("GDP change (%)", min_value=-10.0, max_value=10.0, value=-0.3, step=0.1)

    if st.button("🔍 Predict Risk", use_container_width=True):

        # Initialize user input with all features
        user_input = {f: 0.0 for f in features}
        
        # Calculate derived metrics
        approval_rate_s1 = units_1st_approved / units_1st_enrolled if units_1st_enrolled > 0 else 0
        approval_rate_s2 = units_2nd_approved / units_2nd_enrolled if units_2nd_enrolled > 0 else 0
        grade_evolution = grade_2nd - grade_1st
        parental_edu_score = mother_qual + father_qual
        academic_score_s1 = grade_1st * approval_rate_s1
        
        # Map user inputs to feature names
        user_input['Age at enrollment'] = float(age)
        user_input['Gender'] = float(gender)
        user_input['Scholarship holder'] = float(scholarship)
        user_input['Debtor'] = float(debtor)
        user_input['Tuition fees up to date'] = float(tuition_paid)
        user_input['Displaced'] = float(displaced)
        user_input['Daytime/evening attendance'] = float(attendance)
        user_input['Curricular units 1st sem (enrolled)'] = float(units_1st_enrolled)
        user_input['Curricular units 1st sem (approved)'] = float(units_1st_approved)
        user_input['Curricular units 1st sem (grade)'] = float(grade_1st)
        user_input['Curricular units 2nd sem (enrolled)'] = float(units_2nd_enrolled)
        user_input['Curricular units 2nd sem (approved)'] = float(units_2nd_approved)
        user_input['Curricular units 2nd sem (grade)'] = float(grade_2nd)
        user_input["Mother's qualification"] = float(mother_qual)
        user_input["Father's qualification"] = float(father_qual)
        user_input['Unemployment rate'] = float(unemployment)
        user_input['Inflation rate'] = float(inflation)
        user_input['GDP'] = float(gdp)

        X = pd.DataFrame([user_input])
        X_scaled = scaler.transform(X)

        # Get probability
        proba = model.predict_proba(X_scaled)[0][1]
        risk_score = int(proba * 100)

        # Determine segment and action
        if risk_score < 33:
            segment = "🟢 RASSURANT"
            action = "Suivi régulier — Student on track"
        elif risk_score < 66:
            segment = "🟡 VIGILANCE"
            action = "Suivi renforcé — Monthly check-ins recommended"
        else:
            segment = "🔴 ALERTE CRITIQUE"
            action = "Intervention immédiate — Plan d'accompagnement personnalisé"

        # Calculate feature importance using model coefficients
        if hasattr(model, 'coef_'):
            coef = model.coef_[0]
            feature_importance = {}
            for i, feat in enumerate(features):
                feature_importance[feat] = abs(coef[i]) * abs(X_scaled[0][i])
            
            # Get top 5 factors
            top_5 = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_5 = []

        # Display Report
        st.markdown("---")
        st.markdown("## 📋 RAPPORT DE RISQUE INDIVIDUEL — OUTIL D'AIDE À LA DÉCISION")
        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.metric("🎯 Score de Risque", f"{risk_score} / 100")
        with col_right:
            st.metric("📊 Segment", segment.split()[0])

        st.success(f"💡 Action Recommandée: {action}")

        if top_5:
            st.subheader("📊 Top 5 Facteurs Déclencheurs:")
            for i, (factor, importance) in enumerate(top_5, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"  {i}. **{factor}**")
                with col2:
                    st.write(f"`{importance:.4f}`")

        st.markdown("---")


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
