import streamlit as st 
import pickle 
import numpy as np 

# Load trained models 
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb')) 
heart_disease_model = pickle.load(open('heart_disease_model.pkl', 'rb')) 
parkinsons_model = pickle.load(open('parkinsons_model.pkl', 'rb')) 

# Streamlit page configuration 
st.set_page_config(page_title="Multiple Disease Prediction", layout="centered") 
st.title("Multiple Disease Prediction") 

# Sidebar for disease selection 
st.sidebar.title("Select a Disease") 
disease = st.sidebar.selectbox("Choose a disease to predict:", 
                               ["Diabetes", "Heart Disease", "Parkinson's"]) 

st.subheader(f"Predict {disease}") 

# Input fields based on disease selection 
if disease == "Diabetes": 
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1) 
    glucose = st.number_input("Glucose Level", min_value=0) 
    blood_pressure = st.number_input("Blood Pressure", min_value=0) 
    skin_thickness = st.number_input("Skin Thickness", min_value=0) 
    insulin = st.number_input("Insulin", min_value=0) 
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f") 
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f") 
    age = st.number_input("Age", min_value=0, step=1) 

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]]) 
    model = diabetes_model 

elif disease == "Heart Disease": 
    age = st.number_input("Age", min_value=0, step=1) 
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male 
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3]) 
    trestbps = st.number_input("Resting Blood Pressure", min_value=0) 
    chol = st.number_input("Cholesterol Level", min_value=0) 
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1]) 
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2]) 
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0) 
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1]) 
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, format="%.1f") 
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2]) 
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4]) 
    thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3]) 

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]]) 
    model = heart_disease_model 

elif disease == "Parkinson's": 
    fo = st.number_input("MDVP: Fo(Hz)", min_value=0.0, format="%.2f") 
    fhi = st.number_input("MDVP: Fhi(Hz)", min_value=0.0, format="%.2f") 
    flo = st.number_input("MDVP: Flo(Hz)", min_value=0.0, format="%.2f") 
    jitter_percent = st.number_input("MDVP: Jitter(%)", min_value=0.0, format="%.6f") 
    shimmer = st.number_input("MDVP: Shimmer", min_value=0.0, format="%.6f") 