import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== Load Models ====================
@st.cache_resource
def load_models():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("imputer.pkl", "rb") as f:
            medians = pickle.load(f)
        
        return model, medians
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model files: {e}")
        st.info("Please run the training code first to generate the model files.")
        return None, None

model, medians = load_models()

# ==================== Custom CSS ====================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-card h3 {
        margin-top: 0;
    }
    .result-positive {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
        border: 2px solid #dc3545;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 100%);
        border: 2px solid #28a745;
    }
    .recommendation-box {
        background: #f8f9fa;
        border-left: 4px solid #1a237e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Header ====================
st.markdown("""
    <div class="main-header">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>Early detection can save lives. Enter patient details below for risk assessment.</p>
    </div>
""", unsafe_allow_html=True)

# ==================== Check if models are loaded ====================
if model is None or medians is None:
    st.warning("⚠️ Prediction is disabled until the model files are available. Please run the training code first.")
    st.stop()

# ==================== Input Section ====================
st.markdown("### 📋 Enter Patient Information")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 👤 Personal Information")
    
    pregnancies = st.slider(
        "🤰 Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of times pregnant"
    )
    
    glucose = st.number_input(
        "🩸 Glucose Level (mg/dL)",
        min_value=0,
        max_value=300,
        value=120,
        step=1,
        help="Plasma glucose concentration - Range: 1-300"
    )
    
    blood_pressure = st.number_input(
        "💓 Blood Pressure (mm Hg)",
        min_value=0,
        max_value=200,
        value=70,
        step=1,
        help="Diastolic blood pressure - Range: 1-200"
    )
    
    skin = st.number_input(
        "📏 Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Triceps skin fold thickness - Range: 1-100"
    )

with col2:
    st.markdown("#### 📊 Health Metrics")
    
    insulin = st.number_input(
        "💉 Insulin (mu U/ml)",
        min_value=0,
        max_value=900,
        value=80,
        step=1,
        help="2-Hour serum insulin - Range: 0-900"
    )
    
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        format="%.1f",
        help="Body Mass Index - Range: 0.1-70.0"
    )
    
    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        format="%.3f",
        help="Diabetes pedigree function - Range: 0.001-3.0"
    )
    
    age = st.number_input(
        "🎂 Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        step=1,
        help="Age in years - Range: 1-120"
    )

# ==================== Action Buttons ====================
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_col1, predict_col2 = st.columns(2)
    
    with predict_col1:
        predict_button = st.button("🔮 Predict Diabetes Risk", use_container_width=True, type="primary")
    
    with predict_col2:
        reset_button = st.button("🔄 Reset", use_container_width=True)

# ==================== Reset Functionality ====================
if reset_button:
    st.session_state.clear()
    st.rerun()

# ==================== Prediction Logic ====================
if predict_button:
    # ==================== Validation ====================
    errors = []
    
    if glucose <= 0:
        errors.append("Glucose must be greater than 0.")
    
    if blood_pressure <= 0:
        errors.append("Blood Pressure must be greater than 0.")
    
    if bmi <= 0:
        errors.append("BMI must be greater than 0.")
    
    if age <= 0:
        errors.append("Age must be greater than 0.")
    
    if insulin < 0:
        errors.append("Insulin cannot be negative.")
    
    if skin < 0:
        errors.append("Skin Thickness cannot be negative.")
    
    if dpf <= 0:
        errors.append("Diabetes Pedigree Function must be greater than 0.")
    
    # ==================== Show Validation Errors ====================
    if errors:
        st.error("❌ Please fix the following errors:")
        for error in errors:
            st.markdown(f"- {error}")
        st.stop()
    
    # ==================== Prepare Data ====================
    patient = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age]],
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
    
    # Replace zeros with training medians
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if patient.loc[0, col] == 0:
            patient.loc[0, col] = medians[col]
    
    # ==================== Make Prediction ====================
    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]
    risk_percentage = probability[1] * 100
    
    # ==================== Display Results ====================
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    
    # Two columns for results
    result_col1, result_col2 = st.columns(2, gap="large")
    
    with result_col1:
        # ==================== Result Card ====================
        if prediction == 1:
            st.markdown("""
                <div class="result-card result-positive">
                    <h3>🔴 Diagnosis: Diabetes Detected</h3>
                    <p style="font-size: 1.1rem; color: #dc3545;">
                        The model predicts a high risk of diabetes.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="recommendation-box">
                    <strong>📋 Recommendations:</strong>
                    <ul>
                        <li>Please consult a healthcare professional for confirmatory testing</li>
                        <li>Monitor your blood glucose levels regularly</li>
                        <li>Consider a comprehensive health checkup</li>
                        <li>Discuss lifestyle modifications with your doctor</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-card result-negative">
                    <h3>🟢 Diagnosis: No Diabetes Detected</h3>
                    <p style="font-size: 1.1rem; color: #28a745;">
                        The model predicts a low risk of diabetes.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="recommendation-box">
                    <strong>📋 Recommendations:</strong>
                    <ul>
                        <li>Maintain a healthy lifestyle</li>
                        <li>Continue regular health check-ups</li>
                        <li>Eat a balanced diet and exercise regularly</li>
                        <li>Monitor your health periodically</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # ==================== Risk Classification ====================
        st.markdown("#### 🎯 Risk Classification")
        
        if risk_percentage < 20:
            st.success(f"🟢 Low Risk ({risk_percentage:.1f}%)")
        elif risk_percentage < 40:
            st.info(f"🟡 Mild Risk ({risk_percentage:.1f}%)")
        elif risk_percentage < 60:
            st.warning(f"🟠 Moderate Risk ({risk_percentage:.1f}%)")
        elif risk_percentage < 80:
            st.warning(f"🟧 High Risk ({risk_percentage:.1f}%)")
        else:
            st.error(f"🔴 Very High Risk ({risk_percentage:.1f}%)")
        
        # ==================== Progress Bar ====================
        st.markdown("#### 📊 Risk Level")
        st.progress(risk_percentage / 100)
        
        # ==================== Probability Details ====================
        st.markdown("#### 📈 Probability Details")
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric("Diabetes Risk", f"{risk_percentage:.1f}%")
        
        with col_prob2:
            st.metric("No Diabetes Risk", f"{probability[0]*100:.1f}%")
    
    with result_col2:
        # ==================== Gauge Chart ====================
        st.markdown("#### 🎯 Diabetes Risk Gauge")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            title={'text': "Diabetes Risk (%)", 'font': {'size': 16}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickvals': [0, 20, 40, 60, 80, 100],
                    'ticktext': ['0', '20', '40', '60', '80', '100']
                },
                'bar': {'color': "#1a237e"},
                'steps': [
                    {'range': [0, 20], 'color': '#28a745'},     # Green
                    {'range': [20, 40], 'color': '#ffc107'},    # Yellow
                    {'range': [40, 60], 'color': '#fd7e14'},    # Orange
                    {'range': [60, 80], 'color': '#dc3545'},    # Red
                    {'range': [80, 100], 'color': '#8b0000'}    # Dark Red
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percentage
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ==================== Risk Interpretation ====================
        st.markdown("#### 📝 Risk Interpretation")
        
        if risk_percentage < 20:
            st.info("✅ **Low Risk**: Your risk factors are minimal. Maintain a healthy lifestyle.")
        elif risk_percentage < 40:
            st.info("ℹ️ **Mild Risk**: Some risk factors present. Consider lifestyle improvements.")
        elif risk_percentage < 60:
            st.warning("⚠️ **Moderate Risk**: Several risk factors detected. Consult a healthcare professional.")
        elif risk_percentage < 80:
            st.warning("🚨 **High Risk**: Significant risk factors present. Seek medical advice immediately.")
        else:
            st.error("🔴 **Very High Risk**: Immediate medical attention recommended.")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;">
    🩺 This tool is for educational purposes only. Always consult with a healthcare professional for medical advice.
</div>
""", unsafe_allow_html=True)
