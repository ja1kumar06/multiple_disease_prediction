import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models and accuracies
@st.cache_data
def load_models():
    try:
        with open('models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('accuracies.pkl', 'rb') as f:
            accuracies = pickle.load(f)
        return models, accuracies
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, {}

models, accuracies = load_models()

# App config
st.set_page_config(page_title="Disease Prediction System", page_icon="🧬", layout="wide")
st.title("🧬 ML-Powered Disease Prediction System")
st.markdown("### Powered by Random Forest Models Trained on Synthetic Data")

# Accuracy display
if accuracies:
    combined = sum(accuracies.values()) / len(accuracies)
    cols = st.columns(len(accuracies) + 1)
    for i, key in enumerate(['heart', 'diabetes', 'parkinsons', 'ckd']):
        with cols[i]:
            st.metric(f"{key.title()}", f"{accuracies.get(key, 0):.1%}")
    with cols[-1]:
        st.metric("Combined", f"{combined:.1%}", delta="Target: 90%+")

st.sidebar.title("🔬 Select Disease")
disease = st.sidebar.selectbox("Choose a disease to predict:", ["Heart", "Diabetes", "Parkinsons", "CKD"])

# HEART
if disease == "Heart":
    st.header("❤️ Heart Disease Prediction")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 25, 85, 55)
        sex = st.selectbox("Sex", ["Male (1)", "Female (0.7)"])
        cp = st.selectbox("Chest Pain Type", ["0", "1", "2", "3"])
        trestbps = st.slider("Resting BP", 90, 180, 135)
        chol = st.slider("Cholesterol", 120, 400, 250)
    with col2:
        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        ca = st.selectbox("Major Vessels Colored", ["0", "1", "2", "3"])
        thal = st.selectbox("Thalassemia", ["0", "1", "2", "3"])

    if st.button("🔍 Predict Heart Risk"):
        try:
            input_data = np.array([[age, float(sex.split("(")[1].split(")")[0]), int(cp), trestbps, chol,
                                    int(exang.split("(")[1].split(")")[0]), oldpeak, int(ca), int(thal)]])
            scaled = models['heart']['scaler'].transform(input_data)
            pred = models['heart']['model'].predict(scaled)[0]
            prob = models['heart']['model'].predict_proba(scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ HIGH RISK — Probability: {prob:.1%}")
            else:
                st.success(f"✅ LOW RISK — Probability: {prob:.1%}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# DIABETES
elif disease == "Diabetes":
    st.header("🍯 Diabetes Prediction")
    col1, col2 = st.columns(2)
    with col1:
        preg = st.slider("Pregnancies", 0, 12, 2)
        glucose = st.slider("Glucose", 60, 200, 130)
        bmi = st.slider("BMI", 15.0, 55.0, 33.0)
    with col2:
        age = st.slider("Age", 18, 80, 38)
        dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.8)

    if st.button("🔍 Predict Diabetes Risk"):
        try:
            input_data = np.array([[preg, glucose, bmi, age, dpf]])
            scaled = models['diabetes']['scaler'].transform(input_data)
            pred = models['diabetes']['model'].predict(scaled)[0]
            prob = models['diabetes']['model'].predict_proba(scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ HIGH RISK — Probability: {prob:.1%}")
            else:
                st.success(f"✅ LOW RISK — Probability: {prob:.1%}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# PARKINSONS
elif disease == "Parkinsons":
    st.header("🧠 Parkinson's Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        jitter = st.slider("MDVP_Jitter_percent", 0.002, 0.035, 0.015)
        shimmer = st.slider("MDVP_Shimmer", 0.01, 0.08, 0.04)
    with col2:
        nhr = st.slider("NHR", 0.001, 0.15, 0.03)
        hnr = st.slider("HNR", 8.0, 35.0, 22.0)
    with col3:
        rpde = st.slider("RPDE", 0.25, 0.7, 0.55)
        dfa = st.slider("DFA", 0.57, 0.85, 0.72)
        ppe = st.slider("PPE", 0.04, 0.55, 0.25)

    if st.button("🔍 Predict Parkinson's"):
        try:
            input_data = np.array([[jitter, shimmer, nhr, hnr, rpde, dfa, ppe]])
            scaled = models['parkinsons']['scaler'].transform(input_data)
            pred = models['parkinsons']['model'].predict(scaled)[0]
            prob = models['parkinsons']['model'].predict_proba(scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ INDICATORS DETECTED — Probability: {prob:.1%}")
            else:
                st.success(f"✅ NORMAL — Probability: {prob:.1%}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# CKD
elif disease == "CKD":
    st.header("🫘 CKD Prediction")
    col1, col2 = st.columns(2)
    with col1:
        sc = st.slider("Serum Creatinine", 0.4, 8.0, 1.2)
        bu = st.slider("Blood Urea", 10.0, 150.0, 25.0)
        hemo = st.slider("Hemoglobin", 6.0, 17.0, 12.0)
        pcv = st.slider("Packed Cell Volume", 15, 50, 38)
    with col2:
        al = st.selectbox("Albumin", [0, 1, 2, 3, 4])
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        htn = st.selectbox("Hypertension", ["No (0)", "Yes (1)"])
        dm = st.selectbox("Diabetes Mellitus", ["No (0)", "Yes (1)"])

    if st.button("🔍 Predict CKD Risk"):
        try:
            input_data = np.array([[sc, bu, hemo, pcv, al, float(sg), int(htn.split("(")[1].split(")")[0]), int(dm.split("(")[1].split(")")[0])]])
            scaled = models['ckd']['scaler'].transform(input_data)
            pred = models['ckd']['model'].predict(scaled)[0]
            prob = models['ckd']['model'].predict_proba(scaled)[0][1]
            if pred == 1:
                st.error(f"⚠️ HIGH RISK — Probability: {prob:.1%}")
            else:
                st.success(f"✅ LOW RISK — Probability: {prob:.1%}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("### 🔬 Model Info")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- 5,000 samples per disease\n- Feature engineering\n- Discriminative patterns")
with col2:
    st.markdown("- Random Forest Ensemble\n- Feature scaling\n- Cross-validation")

st.markdown("---")
st.error("⚠️ Medical Disclaimer: This system is for educational use only. Consult professionals for medical decisions.")