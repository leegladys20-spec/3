import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------
# Load Model
# ---------------------------------------
@st.cache_resource
def load_files():
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("imputer.pkl", "rb") as f:
        medians = pickle.load(f)

    return model, medians

model, medians = load_files()

# ---------------------------------------
# Title
# ---------------------------------------
st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict the likelihood of diabetes using a trained Machine Learning model.")

st.divider()

# ---------------------------------------
# Input Form
# ---------------------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.slider(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)",
            min_value=0,
            max_value=200,
            value=70
        )

        skin = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin (mu U/ml)",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.50,
            step=0.01
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    colA, colB = st.columns(2)

    with colA:
        predict = st.form_submit_button("🔍 Predict")

    with colB:
        reset = st.form_submit_button("🔄 Reset")

# ---------------------------------------
# Reset
# ---------------------------------------
if reset:
    st.rerun()

# ---------------------------------------
# Prediction
# ---------------------------------------
if predict:

    errors = []

    if glucose <= 0:
        errors.append("Glucose must be greater than 0.")

    if blood_pressure <= 0:
        errors.append("Blood Pressure must be greater than 0.")

    if bmi <= 0:
        errors.append("BMI must be greater than 0.")

    if age <= 0:
        errors.append("Age must be greater than 0.")

    if dpf <= 0:
        errors.append("Diabetes Pedigree Function must be greater than 0.")

    if errors:

        for e in errors:
            st.error(e)

        st.stop()

    patient = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Replace zero values using training medians
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in zero_columns:
        if patient.loc[0, col] == 0:
            patient.loc[0, col] = medians[col]

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]

    diabetes_prob = probability[1] * 100
    healthy_prob = probability[0] * 100

    st.divider()

    # ---------------------------------------
    # Prediction Result
    # ---------------------------------------
    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("Prediction")

        if prediction == 1:
            st.error("🔴 Diabetes Detected")
        else:
            st.success("🟢 No Diabetes Detected")

        st.metric(
            "Diabetes Probability",
            f"{diabetes_prob:.2f}%"
        )

        st.metric(
            "No Diabetes Probability",
            f"{healthy_prob:.2f}%"
        )

        if diabetes_prob < 20:
            st.success("Risk Level: LOW")

        elif diabetes_prob < 40:
            st.info("Risk Level: MILD")

        elif diabetes_prob < 60:
            st.warning("Risk Level: MODERATE")

        elif diabetes_prob < 80:
            st.warning("Risk Level: HIGH")

        else:
            st.error("Risk Level: VERY HIGH")

    # ---------------------------------------
    # Gauge
    # ---------------------------------------
    with col2:

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=diabetes_prob,

            number={"suffix":"%"},

            title={"text":"Diabetes Risk"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"black"},

                "steps":[

                    {"range":[0,20],"color":"green"},

                    {"range":[20,40],"color":"yellowgreen"},

                    {"range":[40,60],"color":"gold"},

                    {"range":[60,80],"color":"orange"},

                    {"range":[80,100],"color":"red"}

                ]

            }

        ))

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------
    # Patient Summary
    # ---------------------------------------
    st.subheader("Patient Information")

    st.dataframe(patient, use_container_width=True)

    st.divider()

    st.subheader("Recommendation")

    if prediction == 1:

        st.warning("""
The model predicts an elevated likelihood of diabetes.

**Recommendation**
- Consult a healthcare professional.
- Consider confirmatory laboratory testing.
- Maintain a healthy diet.
- Exercise regularly.
- Monitor blood glucose as advised.
""")

    else:

        st.success("""
The model predicts a lower likelihood of diabetes.

**Recommendation**
- Maintain a healthy lifestyle.
- Exercise regularly.
- Eat a balanced diet.
- Continue regular health check-ups.
""")
