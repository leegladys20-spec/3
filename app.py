import streamlit as st
import pickle
import pandas as pd

# Load model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load medians
with open("imputer.pkl", "rb") as f:
    medians = pickle.load(f)

st.title("Diabetes Prediction")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):

    patient = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                             skin, insulin, bmi, dpf, age]],
                           columns=[
                               "Pregnancies",
                               "Glucose",
                               "BloodPressure",
                               "SkinThickness",
                               "Insulin",
                               "BMI",
                               "DiabetesPedigreeFunction",
                               "Age"
                           ])

    # Replace zeros with training medians
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if patient.loc[0, col] == 0:
            patient.loc[0, col] = medians[col]

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]

    if prediction == 1:
        st.error("Diabetes Detected")
    else:
        st.success("No Diabetes Detected")

    st.write(f"Probability of Diabetes: **{probability[1]*100:.2f}%**")
    st.write(f"Probability of No Diabetes: **{probability[0]*100:.2f}%**")
