import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
import io

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# Load Model with Error Handling
# =====================================================
@st.cache_resource
def load_model():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("imputer.pkl", "rb") as f:
            medians = pickle.load(f)
        
        return model, medians
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, medians = load_model()

# =====================================================
# Custom CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: #EEF4FF;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 30px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: bold;
    color: #1a237e;
}

.main-title {
    text-align: center;
    color: #1A237E;
    font-size: 48px;
    font-weight: 800;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 40px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,.08);
}

.section {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 15px;
}

.info {
    background: #dbeafe;
    padding: 18px;
    border-radius: 12px;
    color: #0f172a;
    font-size: 17px;
}

div.stButton > button {
    width: 100%;
    background: #1A237E;
    color: white;
    border: none;
    border-radius: 30px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: #283593;
    color: white;
}

div.stForm button {
    background: #1A237E;
    color: white;
    border-radius: 30px;
    font-weight: bold;
}

[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Helper Functions
# =====================================================
def validate_required_fields(glucose, blood_pressure, bmi, age, dpf):
    """Validate required fields for prediction"""
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
    
    return errors

def replace_zero_values(df, columns):
    """Replace zero values with median values"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(0, medians.get(col, 0))
    return df

def create_gauge_chart(diabetes_prob):
    """Create a gauge chart for diabetes risk"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=diabetes_prob,
            number={"suffix": "%"},
            title={"text": "Diabetes Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1A237E"},
                "steps": [
                    {"range": [0, 20], "color": "#4CAF50"},
                    {"range": [20, 40], "color": "#8BC34A"},
                    {"range": [40, 60], "color": "#FFC107"},
                    {"range": [60, 80], "color": "#FF9800"},
                    {"range": [80, 100], "color": "#F44336"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.8,
                    "value": diabetes_prob
                }
            }
        )
    )
    
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def display_recommendation(prediction):
    """Display health recommendations based on prediction"""
    if prediction == 1:
        st.warning("""
        ### ⚠️ High Risk Detected
        
        The machine learning model predicts an elevated likelihood of diabetes.
        
        ### 📋 Recommended Actions
        - Consult a healthcare professional immediately
        - Schedule comprehensive laboratory testing
        - Monitor blood glucose regularly
        - Follow a balanced, low-sugar diet
        - Exercise for at least 30 minutes daily
        - Maintain a healthy weight
        - Attend regular medical checkups
        """)
    else:
        st.success("""
        ### ✅ Low Risk Detected
        
        The model predicts a lower likelihood of diabetes.
        
        ### 📋 Recommended Actions
        - Continue a balanced and nutritious diet
        - Exercise regularly (30+ minutes/day)
        - Stay hydrated
        - Maintain a healthy weight
        - Get annual health checkups
        - Practice healthy lifestyle habits
        """)

# =====================================================
# Navigation
# =====================================================
tab1, tab2 = st.tabs([
    "🩺 Diabetes Prediction",
    "⚖️ BMI Calculator"
])

# =====================================================
# BMI Calculator
# =====================================================
with tab2:
    st.title("⚖️ BMI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=70.0,
            step=0.1
        )
    
    with col2:
        height = st.number_input(
            "Height (m)",
            min_value=0.50,
            max_value=2.50,
            value=1.70,
            step=0.01
        )
    
    if st.button("Calculate BMI", use_container_width=True):
        bmi = weight / (height ** 2)
        
        st.metric("BMI", f"{bmi:.2f}")
        
        if bmi < 18.5:
            st.info("📉 Underweight - Consider consulting a nutritionist")
        elif bmi < 25:
            st.success("✅ Normal Weight - Keep up the good work!")
        elif bmi < 30:
            st.warning("⚠️ Overweight - Consider lifestyle changes")
        else:
            st.error("❌ Obese - Please consult a healthcare professional")

# =====================================================
# Diabetes Prediction
# =====================================================
with tab1:
    st.markdown(
        "<h1 class='main-title'>🩺 Diabetes Prediction System</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p class='sub-title'>Early detection can save lives. Enter patient details below for risk assessment.</p>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='section'>📋 Select Input Method</div>",
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        manual = st.button("✏️ Manual Input", use_container_width=True)
    
    with col2:
        upload = st.button("📁 Upload CSV", use_container_width=True)
    
    if "mode" not in st.session_state:
        st.session_state.mode = "manual"
    
    if manual:
        st.session_state.mode = "manual"
    
    if upload:
        st.session_state.mode = "upload"
    
    # Display mode indicator
    if st.session_state.mode == "manual":
        st.markdown("""
        <div class="info">
        📋 Currently using <b>Manual Input</b> mode. Enter patient details below.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info">
        📁 Currently using <b>CSV Upload</b> mode. Upload a CSV file with patient data.
        </div>
        """, unsafe_allow_html=True)
    
    # =====================================================
    # MANUAL INPUT
    # =====================================================
    if st.session_state.mode == "manual":
        with st.form("prediction_form"):
            left, right = st.columns(2)
            
            with left:
                pregnancies = st.slider(
                    "👶 Pregnancies",
                    0,
                    20,
                    1,
                    help="Number of pregnancies"
                )
                
                glucose = st.number_input(
                    "🩸 Glucose (mg/dL)",
                    min_value=0,
                    max_value=300,
                    value=120,
                    help="Glucose level in blood"
                )
                
                blood_pressure = st.number_input(
                    "❤️ Blood Pressure (mmHg)",
                    min_value=0,
                    max_value=200,
                    value=70,
                    help="Diastolic blood pressure"
                )
                
                skin = st.number_input(
                    "📏 Skin Thickness (mm)",
                    min_value=0,
                    max_value=100,
                    value=20,
                    help="Triceps skin fold thickness"
                )
            
            with right:
                insulin = st.number_input(
                    "💉 Insulin (mu U/ml)",
                    min_value=0,
                    max_value=900,
                    value=80,
                    help="2-Hour serum insulin"
                )
                
                bmi = st.number_input(
                    "⚖️ BMI",
                    min_value=0.0,
                    max_value=70.0,
                    value=25.0,
                    step=0.1,
                    help="Body Mass Index"
                )
                
                dpf = st.number_input(
                    "📊 Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.50,
                    step=0.01,
                    help="Diabetes pedigree function"
                )
                
                age = st.number_input(
                    "🎂 Age",
                    min_value=1,
                    max_value=120,
                    value=30,
                    help="Age in years"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                predict = st.form_submit_button("🔍 Predict Diabetes", use_container_width=True)
            
            with col2:
                reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)
        
        if reset:
            st.rerun()
        
        if predict:
            # Validate inputs
            errors = validate_required_fields(glucose, blood_pressure, bmi, age, dpf)
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                st.stop()
            
            # Create patient dataframe
            patient = pd.DataFrame(
                [[pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age]],
                columns=[
                    "Pregnancies", "Glucose", "BloodPressure", 
                    "SkinThickness", "Insulin", "BMI", 
                    "DiabetesPedigreeFunction", "Age"
                ]
            )
            
            # Replace zero values with medians
            zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            patient = replace_zero_values(patient, zero_columns)
            
            # Make prediction
            try:
                prediction = model.predict(patient)[0]
                probability = model.predict_proba(patient)[0]
                
                diabetes_prob = probability[1] * 100
                healthy_prob = probability[0] * 100
                
                st.session_state.prediction = prediction
                st.session_state.patient = patient
                st.session_state.diabetes_prob = diabetes_prob
                st.session_state.healthy_prob = healthy_prob
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # =====================================================
    # CSV Upload
    # =====================================================
    if st.session_state.mode == "upload":
        uploaded_file = st.file_uploader(
            "📤 Upload CSV File",
            type=["csv"],
            help="Upload a CSV file with the required columns"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = [
                    "Pregnancies", "Glucose", "BloodPressure", 
                    "SkinThickness", "Insulin", "BMI", 
                    "DiabetesPedigreeFunction", "Age"
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                else:
                    st.subheader("📊 Uploaded Data")
                    st.dataframe(df, use_container_width=True)
                    
                    if st.button("🚀 Predict Uploaded Data", use_container_width=True):
                        with st.spinner("Making predictions..."):
                            # Replace zero values
                            zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                            df = replace_zero_values(df, zero_columns)
                            
                            try:
                                # Make predictions
                                predictions = model.predict(df)
                                probabilities = model.predict_proba(df)
                                
                                # Add results to dataframe
                                df["Prediction"] = predictions
                                df["Diabetes_Probability"] = (probabilities[:, 1] * 100).round(2)
                                df["Risk_Level"] = pd.cut(
                                    df["Diabetes_Probability"],
                                    bins=[0, 20, 40, 60, 80, 100],
                                    labels=["Low", "Mild", "Moderate", "High", "Very High"]
                                )
                                
                                st.success("✅ Prediction completed successfully!")
                                st.dataframe(df, use_container_width=True)
                                
                                # Download results
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "💾 Download Results",
                                    csv,
                                    "predictions.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"Error making predictions: {e}")
                            
            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty. Please upload a valid CSV file.")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    # =====================================================
    # SHOW PREDICTION RESULT
    # =====================================================
    if "prediction" in st.session_state:
        st.markdown("---")
        
        prediction = st.session_state.prediction
        patient = st.session_state.patient
        diabetes_prob = st.session_state.diabetes_prob
        healthy_prob = st.session_state.healthy_prob
        
        col1, col2 = st.columns([1, 1])
        
        # Prediction Summary
        with col1:
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:
                st.error("🔴 **Diabetes Detected**")
            else:
                st.success("🟢 **No Diabetes Detected**")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Diabetes Probability", f"{diabetes_prob:.2f}%")
            with col_b:
                st.metric("Healthy Probability", f"{healthy_prob:.2f}%")
            
            # Risk Level
            if diabetes_prob < 20:
                st.success("🟢 Risk Level: LOW")
            elif diabetes_prob < 40:
                st.info("🟡 Risk Level: MILD")
            elif diabetes_prob < 60:
                st.warning("🟠 Risk Level: MODERATE")
            elif diabetes_prob < 80:
                st.warning("🔶 Risk Level: HIGH")
            else:
                st.error("🔴 Risk Level: VERY HIGH")
        
        # Gauge Chart
        with col2:
            fig = create_gauge_chart(diabetes_prob)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Patient Information
        st.subheader("👤 Patient Information")
        st.dataframe(patient, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        display_recommendation(prediction)
