"""
Streamlit app for Heart Disease Prediction.
Run: streamlit run app_streamlit.py
It loads best_model.joblib created by notebook_train.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

st.title("Heart Disease Risk Predictor")
st.write("Enter patient data below to get a risk prediction.")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.joblib')
        return model
    except Exception as e:
        st.error("Could not load best_model.joblib. Run training script first.")
        return None

model = load_model()
if model is None:
    st.stop()

# Input sidebar or form
with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.selectbox("Chest pain type (cp)", options=[0,1,2,3])
    trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum cholesterol (chol)", min_value=100, max_value=600, value=250)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0,1])
    restecg = st.selectbox("Resting ECG results (restecg)", options=[0,1,2])
    thalach = st.number_input("Max heart rate achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-induced angina (exang)", options=[0,1])
    oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", options=[0,1,2])
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", options=[0,1,2,3,4])
    thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", options=[1,2,3])
    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    proba = model.predict_proba(X_new)[0,1] if hasattr(model, "predict_proba") else None
    pred = model.predict(X_new)[0]
    st.subheader("Prediction")
    if pred == 1:
        st.error(f"Model prediction: HIGH RISK of heart disease (probability: {proba:.2f})")
    else:
        st.success(f"Model prediction: LOW RISK of heart disease (probability: {proba:.2f})")
    st.write("**Note:** This tool is a decision-support prototype. Consult a clinician for diagnosis.")
