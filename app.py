# ============================================
# Multi-Disease Prediction App using Streamlit
# ============================================

import streamlit as st
import numpy as np
import joblib

# --------------------------
# Load trained components
# --------------------------
model = joblib.load("multi_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
le_disease = joblib.load("label_encoder.pkl")
le_gender = joblib.load("gender_encoder.pkl")

# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠", layout="centered")

st.title("🩺 AI-Based Disease Prediction System")
st.write("Enter your health details below to predict possible diseases.")

# --------------------------
# Collect user input
# --------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
    glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 100)

with col2:
    chol = st.number_input("Cholesterol Level (mg/dL)", 100, 400, 180)
    liver = st.number_input("Liver Enzyme Level", 10.0, 200.0, 40.0)
    creatinine = st.number_input("Creatinine Level (mg/dL)", 0.2, 5.0, 1.0)
    hemoglobin = st.number_input("Hemoglobin Level (g/dL)", 5.0, 20.0, 14.0)
    oxygen = st.number_input("Oxygen Level (%)", 70.0, 100.0, 97.0)

col3, col4 = st.columns(2)

with col3:
    crp = st.number_input("CRP Level (mg/L)", 0.1, 50.0, 5.0)
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol = st.selectbox("Do you consume alcohol?", ["No", "Yes"])

with col4:
    exercise = st.slider("Exercise per week (times)", 0, 10, 3)
    family = st.selectbox("Family History of Disease?", ["No", "Yes"])

# --------------------------
# Process and predict
# --------------------------
if st.button("🔍 Predict Disease"):
    try:
        # Encode gender and yes/no values
        gender_encoded = le_gender.transform([gender])[0]
        smoking_val = 1 if smoking == "Yes" else 0
        alcohol_val = 1 if alcohol == "Yes" else 0
        family_val = 1 if family == "Yes" else 0

        # Create feature array
        input_data = np.array([
            [
                age, gender_encoded, bmi, bp, glucose, chol, liver,
                creatinine, hemoglobin, oxygen, crp, smoking_val,
                alcohol_val, exercise, family_val
            ]
        ])

        # Scale the input
        scaled_input = scaler.transform(input_data)

        # Predict
        pred = model.predict(scaled_input)[0]
        disease_name = le_disease.inverse_transform([pred])[0]

        # Get prediction probabilities
        probs = model.predict_proba(scaled_input)[0]
        top_indices = np.argsort(probs)[::-1][:3]  # top 3 predictions
        top_diseases = le_disease.inverse_transform(top_indices)
        top_probs = probs[top_indices] * 100

        st.success(f"### 🧠 Prediction: **{disease_name}**")
        st.write("**Top possible conditions:**")

        for d, p in zip(top_diseases, top_probs):
            st.write(f"- {d}: {p:.2f}% confidence")

    except Exception as e:
        st.error(f"Error: {e}")
