import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model, scaler, and column names
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')


st.title("Heart Disease Prediction")
st.markdown("Enter the  details to predict the heart disease:")


    
age = st.slider("Age", 18, 100 , 54)
sex = st.selectbox("Sex", options=["Male", "Female"])
cheast_pain = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_BP = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 240)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ECG = st.selectbox("Resting ECG", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", [0,1])
st_slope = st.selectbox("ST Slope", options=["Upsloping", "Flat", "Downsloping"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

if st.button("Predict"):
    # Create a DataFrame for the input data using the same feature names as the trained model
    input_data = pd.DataFrame([{
        'Age': age,
        'RestingBP': resting_BP,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_blood_sugar,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == "Male" else 0,
        'ChestPainType_TA': 1 if cheast_pain == "Typical Angina" else 0,
        'ChestPainType_ATA': 1 if cheast_pain == "Atypical Angina" else 0,
        'ChestPainType_NAP': 1 if cheast_pain == "Non-Anginal Pain" else 0,
        'RestingECG_Normal': 1 if resting_ECG == "Normal" else 0,
        'RestingECG_ST': 1 if resting_ECG == "ST-T Wave Abnormality" else 0,
        'ExerciseAngina_Y': 1 if exercise_induced_angina == 1 else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Upsloping" else 0
    }])
    # Align input columns with the trained model's columns and scale the data
    input_data = input_data.reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    # Display the result
    if prediction[0] == 1:
        st.error("The model predicts that you have heart disease.")
    else:
        st.success("The model predicts that you do not have heart disease.")