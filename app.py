import streamlit as st
import pickle
import numpy as np

# Load trained models
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinson_model.pkl', 'rb'))
breast_cancer= pickle.load(open('breast_cancer.pkl', 'rb'))

# Streamlit page configuration
st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")
st.title("Multiple Disease Prediction")

# Sidebar for disease selection
st.sidebar.title("Select a Disease")
disease = st.sidebar.selectbox("Choose a disease to predict:", 
                               ["Diabetes", "Heart Disease", "Parkinson's", "Breast Cancer"])

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
    model = heart_model

elif disease == "Parkinson's":
    fo = st.number_input("MDVP: Fo(Hz)", min_value=0.0, format="%.2f")
    fhi = st.number_input("MDVP: Fhi(Hz)", min_value=0.0, format="%.2f")
    flo = st.number_input("MDVP: Flo(Hz)", min_value=0.0, format="%.2f")
    jitter_percent = st.number_input("MDVP: Jitter(%)", min_value=0.0, format="%.6f")
    jitter_abs = st.number_input("MDVP: Jitter(Abs)", min_value=0.0, format="%.6f")
    shimmer = st.number_input("MDVP: Shimmer", min_value=0.0, format="%.6f")
    shimmer_dB = st.number_input("MDVP: Shimmer(dB)", min_value=0.0, format="%.2f")
    nhr = st.number_input("NHR", min_value=0.0, format="%.6f")
    hnr = st.number_input("HNR", min_value=0.0, format="%.2f")
    rpde = st.number_input("RPDE", min_value=0.0, format="%.6f")
    dfa = st.number_input("DFA", min_value=0.0, format="%.6f")
    spread1 = st.number_input("Spread1", format="%.6f")
    spread2 = st.number_input("Spread2", format="%.6f")
    d2 = st.number_input("D2", min_value=0.0, format="%.6f")
    ppe = st.number_input("PPE", min_value=0.0, format="%.6f")

    input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, shimmer, shimmer_dB, 
                            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
    model = parkinsons_model

elif disease == "Breast Cancer":
    radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.2f")
    texture_mean = st.number_input("Texture Mean", min_value=0.0, format="%.2f")
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.2f")
    area_mean = st.number_input("Area Mean", min_value=0.0, format="%.2f")
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, format="%.6f")
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, format="%.6f")
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, format="%.6f")
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, format="%.6f")
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, format="%.6f")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, format="%.6f")

    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, 
                            smoothness_mean, compactness_mean, concavity_mean, 
                            concave_points_mean, symmetry_mean, fractal_dimension_mean]])
    model = breast_cancer_model

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Positive (Disease Detected)" if prediction == 1 else "Negative (No Disease)"
    st.write(f"Prediction Result: **{result}**")
