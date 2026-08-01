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
    background: #f5f7fa;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 30px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: bold;
    color: #1a237e;
}

/* Main Title */
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

/* Normal Button Styles */
div.stButton > button {
    width: 100%;
    background: #f0f2f6;
    color: #1a1a1a;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    height: 50px;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #e4e7ec;
    border-color: #b0b5bd;
    color: #1a1a1a;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* Form Button */
div.stForm button {
    background: #f0f2f6;
    color: #1a1a1a;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    height: 50px;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

div.stForm button:hover {
    background: #e4e7ec;
    border-color: #b0b5bd;
    color: #1a1a1a;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* ============================================= */
/* BMI Calculator - Bigger Inputs */
/* ============================================= */
.bmi-input-container {
    background: white;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.bmi-input-container .stNumberInput {
    width: 100%;
}

.bmi-input-container .stNumberInput input {
    font-size: 24px !important;
    padding: 20px 15px !important;
    height: 70px !important;
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
    transition: all 0.3s ease;
}

.bmi-input-container .stNumberInput input:focus {
    border-color: #1A237E !important;
    box-shadow: 0 0 0 3px rgba(26, 35, 126, 0.1) !important;
}

.bmi-input-container .stNumberInput label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #333 !important;
    margin-bottom: 8px !important;
}

.bmi-calculate-btn {
    margin-top: 20px;
}

.bmi-calculate-btn button {
    width: 100% !important;
    height: 60px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    background: #1A237E !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.bmi-calculate-btn button:hover {
    background: #283593 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(26, 35, 126, 0.3) !important;
}

.bmi-calculate-btn button:active {
    transform: translateY(0px) !important;
}

/* ============================================= */
/* Slider Styling - Dark Blue */
/* ============================================= */
div[data-baseweb="slider"] {
    margin-top: 5px;
}

div[data-baseweb="slider"] div[role="slider"] {
    background: #1A237E !important;
    width: 18px !important;
    height: 18px !important;
    border: 2px solid white !important;
    box-shadow: 0 2px 6px rgba(26, 35, 126, 0.3) !important;
}

div[data-baseweb="slider"] div[data-testid="stSliderTrack"] {
    background: #e0e0e0 !important;
    height: 6px !important;
    border-radius: 3px !important;
}

div[data-baseweb="slider"] div[data-testid="stSliderTrack"] > div {
    background: #1A237E !important;
}

/* ============================================= */
/* Number Input - No Box Around +/- Buttons */
/* ============================================= */
div[data-testid="stNumberInput"] {
    position: relative;
}

div[data-testid="stNumberInput"] button {
    background: transparent !important;
    color: #1A237E !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 4px 8px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    min-width: 30px !important;
    min-height: 30px !important;
    box-shadow: none !important;
    transition: all 0.2s ease;
}

div[data-testid="stNumberInput"] button:hover {
    background: rgba(26, 35, 126, 0.08) !important;
    color: #1A237E !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stNumberInput"] button:active {
    background: rgba(26, 35, 126, 0.15) !important;
    transform: scale(0.95);
}

div[data-testid="stNumberInput"] button:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    border: 2px solid #d0d5dd !important;
    border-radius: 8px !important;
    background: white !important;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: #1A237E !important;
    box-shadow: 0 0 0 2px rgba(26, 35, 126, 0.1) !important;
}

/* ============================================= */
/* BMI Result Styles */
/* ============================================= */
.bmi-result-box {
    background: white;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.bmi-value-large {
    font-size: 56px;
    font-weight: 800;
    color: #1a1a1a;
    line-height: 1;
    margin: 10px 0 5px 0;
}

.bmi-category {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}

.bmi-message {
    font-size: 16px;
    color: #555;
}

.bmi-scale-container {
    background: white;
    border-radius: 16px;
    padding: 25px 30px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin: 20px 0;
}

.bmi-scale-bar {
    position: relative;
    height: 30px;
    border-radius: 15px;
    background: linear-gradient(to right, #4fc3f7, #81c784, #fff176, #ff8a65, #ef5350);
    margin: 20px 0 30px 0;
    overflow: visible;
}

.bmi-marker {
    position: absolute;
    top: -12px;
    transform: translateX(-50%);
    width: 28px;
    height: 28px;
    background: #1a237e;
    border: 3px solid white;
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 10;
    transition: left 0.5s ease;
}

.bmi-labels {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #555;
    padding: 0 5px;
    margin-top: 5px;
}

.bmi-labels span {
    text-align: center;
    flex: 1;
}

.bmi-info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin: 20px 0;
}

.bmi-info-item {
    background: white;
    padding: 18px 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border-left: 4px solid #1a237e;
}

.bmi-info-item .label {
    font-size: 13px;
    color: #888;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.bmi-info-item .value {
    font-size: 20px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 4px;
}

.bmi-category-item {
    display: flex;
    justify-content: space-between;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 5px 0;
    font-size: 14px;
}

.bmi-category-item .range {
    color: #666;
    font-size: 13px;
}

.bmi-category-item.active {
    background: #e8eaf6;
    font-weight: 600;
    border-left: 4px solid #1a237e;
}

.bmi-category-item.underweight { border-left: 4px solid #4fc3f7; }
.bmi-category-item.normal { border-left: 4px solid #81c784; }
.bmi-category-item.overweight { border-left: 4px solid #fff176; }
.bmi-category-item.obese { border-left: 4px solid #ef5350; }

.bmi-note {
    background: #f8f9fa;
    padding: 15px 20px;
    border-radius: 10px;
    font-size: 14px;
    color: #666;
    margin-top: 20px;
    border-left: 4px solid #1a237e;
}

/* File Uploader */
div[data-testid="stFileUploader"] button {
    background: #f0f2f6 !important;
    color: #1a1a1a !important;
    border: 1px solid #d0d5dd !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #e4e7ec !important;
    border-color: #b0b5bd !important;
}

/* Download Button */
div[data-testid="stDownloadButton"] button {
    background: #f0f2f6 !important;
    color: #1a1a1a !important;
    border: 1px solid #d0d5dd !important;
    border-radius: 8px !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background: #e4e7ec !important;
    border-color: #b0b5bd !important;
}

/* Responsive */
@media (max-width: 768px) {
    .bmi-info-grid {
        grid-template-columns: 1fr;
    }
    .bmi-value-large {
        font-size: 40px;
    }
    .bmi-input-container .stNumberInput input {
        font-size: 18px !important;
        height: 55px !important;
        padding: 15px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Helper Functions
# =====================================================
def validate_required_fields(glucose, blood_pressure, bmi, age, dpf, skin, insulin, pregnancies):
    """Validate required fields for prediction"""
    errors = []
    
    if pregnancies < 0 or pregnancies > 20:
        errors.append("Pregnancies must be between 0 and 20.")
    
    if glucose <= 0 or glucose > 300:
        errors.append("Glucose must be between 1 and 300 mg/dL.")
    
    if blood_pressure <= 0 or blood_pressure > 200:
        errors.append("Blood Pressure must be between 1 and 200 mmHg.")
    
    if skin < 0.5 or skin > 4.0:
        errors.append("Skin Thickness must be between 0.5 and 4.0 mm.")
    
    if insulin < 0 or insulin > 900:
        errors.append("Insulin must be between 0 and 900 mu U/ml.")
    
    if bmi <= 0 or bmi > 100:
        errors.append("BMI must be between 0.1 and 100 kg/m².")
    
    if age < 1 or age > 120:
        errors.append("Age must be between 1 and 120 years.")
    
    if dpf <= 0 or dpf > 3:
        errors.append("Diabetes Pedigree Function must be between 0.01 and 3.0.")
    
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
# BMI Calculator Component - Redesigned
# =====================================================
def bmi_calculator():
    """BMI Calculator with bigger inputs and full-width layout"""
    
    st.markdown("<h1 style='text-align: center; color: #1A237E;'>⚖️ BMI Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555; margin-bottom: 30px;'>Calculate your Body Mass Index and assess your health status</p>", unsafe_allow_html=True)
    
    # Input Container - Full Width with bigger inputs
    st.markdown('<div class="bmi-input-container">', unsafe_allow_html=True)
    
    # Weight and Height in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=70.0,
            step=0.5,
            help="Enter your weight in kilograms",
            key="bmi_weight"
        )
    
    with col2:
        height = st.number_input(
            "Height (m)",
            min_value=0.50,
            max_value=2.50,
            value=1.70,
            step=0.01,
            help="Enter your height in meters",
            key="bmi_height"
        )
    
    # Calculate Button - Full width below inputs
    st.markdown('<div class="bmi-calculate-btn">', unsafe_allow_html=True)
    calculate_clicked = st.button("📊 Calculate BMI", use_container_width=True, key="bmi_calculate")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results if calculated
    if calculate_clicked:
        bmi = weight / (height ** 2)
        
        # Determine category
        if bmi < 18.5:
            category = "Underweight"
            color = "#4fc3f7"
            emoji = "📉"
            message = "Consider consulting a nutritionist for a healthy weight gain plan."
            position = (bmi / 40) * 100
        elif bmi < 25:
            category = "Normal Weight"
            color = "#66bb6a"
            emoji = "✅"
            message = "Great job! Maintain your healthy lifestyle."
            position = ((bmi - 18.5) / (24.9 - 18.5)) * 25 + 25
        elif bmi < 30:
            category = "Overweight"
            color = "#ffca28"
            emoji = "⚠️"
            message = "Consider lifestyle changes to reach a healthy weight."
            position = ((bmi - 25) / (29.9 - 25)) * 25 + 50
        else:
            category = "Obese"
            color = "#ef5350"
            emoji = "❌"
            message = "Please consult a healthcare professional for guidance."
            position = min(((bmi - 30) / 10) * 25 + 75, 95)
        
        position = max(2, min(98, position))
        
        # BMI Result Display
        st.markdown(f"""
        <div class="bmi-result-box">
            <div style="font-size: 16px; color: #888; font-weight: 500;">Your BMI</div>
            <div class="bmi-value-large">{bmi:.1f}</div>
            <div class="bmi-category" style="color: {color};">{emoji} {category}</div>
            <div class="bmi-message">{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # BMI Scale Bar
        st.markdown("""
        <div class="bmi-scale-container">
            <div style="text-align: center; font-weight: 600; font-size: 18px; margin-bottom: 10px;">
                BMI Scale
            </div>
            <div class="bmi-scale-bar">
                <div class="bmi-marker" style="left: {:.1f}%;"></div>
            </div>
            <div class="bmi-labels">
                <span style="color: #4fc3f7;">Underweight</span>
                <span style="color: #81c784;">Normal</span>
                <span style="color: #fff176;">Overweight</span>
                <span style="color: #ef5350;">Obese</span>
            </div>
        </div>
        """.format(position), unsafe_allow_html=True)
        
        # Detailed Information
        st.markdown("### 📋 Detailed BMI Information")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="bmi-info-grid">
                <div class="bmi-info-item">
                    <div class="label">Your BMI</div>
                    <div class="value">{bmi:.1f}</div>
                </div>
                <div class="bmi-info-item" style="border-left-color: {color};">
                    <div class="label">Category</div>
                    <div class="value" style="color: {color};">{category}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            categories = [
                ("Underweight", "< 18.5", bmi < 18.5, "underweight"),
                ("Normal", "18.5 - 24.9", 18.5 <= bmi < 25, "normal"),
                ("Overweight", "25 - 29.9", 25 <= bmi < 30, "overweight"),
                ("Obese", ">= 30", bmi >= 30, "obese")
            ]
            
            for name, range_text, active, class_name in categories:
                active_class = "active" if active else ""
                st.markdown(f"""
                <div class="bmi-category-item {class_name} {active_class}">
                    <span>{name}</span>
                    <span class="range">{range_text}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Health Implications
        st.markdown("### 💡 Health Implications")
        
        if bmi < 18.5:
            st.info("""
            **Underweight (< 18.5):** May indicate malnutrition, eating disorders, 
            or other health issues. Consider consulting a healthcare provider.
            """)
        elif bmi < 25:
            st.success("""
            **Normal (18.5 - 24.9):** Healthy weight range for most adults. 
            Keep up the good work with a balanced diet and regular exercise.
            """)
        elif bmi < 30:
            st.warning("""
            **Overweight (25 - 29.9):** Increased risk of health problems. 
            Consider adopting healthier eating habits and increasing physical activity.
            """)
        else:
            st.error("""
            **Obese (>= 30):** High risk of health problems including diabetes, 
            heart disease, and more. Please consult a healthcare professional.
            """)
        
        st.markdown("""
        <div class="bmi-note">
            <strong>📌 Note:</strong> BMI is a screening tool and doesn't account for 
            muscle mass, bone density, or overall body composition. It should be used 
            as a general guideline, not a definitive diagnostic tool.
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# Navigation
# =====================================================
tab1, tab2 = st.tabs([
    "🩺 Diabetes Prediction",
    "⚖️ BMI Calculator"
])

# =====================================================
# BMI Calculator Tab
# =====================================================
with tab2:
    bmi_calculator()

# =====================================================
# Diabetes Prediction Tab
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
        # Clear any previous prediction results when switching modes
        if "prediction" in st.session_state:
            del st.session_state.prediction
            del st.session_state.patient
            del st.session_state.diabetes_prob
            del st.session_state.healthy_prob
    
    if upload:
        st.session_state.mode = "upload"
        # Clear any previous prediction results when switching modes
        if "prediction" in st.session_state:
            del st.session_state.prediction
            del st.session_state.patient
            del st.session_state.diabetes_prob
            del st.session_state.healthy_prob
    
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
        # Initialize session state for form values if not exists
        if "form_values" not in st.session_state:
            st.session_state.form_values = {
                "pregnancies": 0,
                "glucose": 0,
                "blood_pressure": 0,
                "skin": 0.5,  # Set to minimum valid value
                "insulin": 0,
                "bmi": 0.0,
                "dpf": 0.0,
                "age": 0
            }
        
        with st.form("prediction_form"):
            left, right = st.columns(2)
            
            with left:
                pregnancies = st.slider(
                    "👶 Pregnancies",
                    0,
                    20,
                    value=st.session_state.form_values["pregnancies"],
                    help="Number of pregnancies"
                )
                
                glucose = st.number_input(
                    "🩸 Glucose (mg/dL)",
                    min_value=0,
                    max_value=300,
                    value=st.session_state.form_values["glucose"],
                    help="Glucose level in blood"
                )
                
                blood_pressure = st.number_input(
                    "❤️ Blood Pressure (mmHg)",
                    min_value=0,
                    max_value=200,
                    value=st.session_state.form_values["blood_pressure"],
                    help="Diastolic blood pressure"
                )
                
                skin = st.number_input(
                    "📏 Skin Thickness (mm)",
                    min_value=0.5,
                    max_value=4.0,
                    value=st.session_state.form_values["skin"],
                    step=0.1,
                    help="Triceps skin fold thickness (0.5 - 4.0 mm)"
                )
            
            with right:
                insulin = st.number_input(
                    "💉 Insulin (mu U/ml)",
                    min_value=0,
                    max_value=900,
                    value=st.session_state.form_values["insulin"],
                    help="2-Hour serum insulin"
                )
                
                bmi = st.number_input(
                    "⚖️ BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.form_values["bmi"],
                    step=0.1,
                    help="Body Mass Index"
                )
                
                dpf = st.number_input(
                    "📊 Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=st.session_state.form_values["dpf"],
                    step=0.01,
                    help="Diabetes pedigree function"
                )
                
                age = st.number_input(
                    "🎂 Age",
                    min_value=0,
                    max_value=120,
                    value=st.session_state.form_values["age"],
                    help="Age in years"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                predict = st.form_submit_button("🔍 Predict Diabetes", use_container_width=True)
            
            with col2:
                reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)
        
        # Handle Reset - Reset all form values to defaults
        if reset:
            # Reset form values in session state
            st.session_state.form_values = {
                "pregnancies": 0,
                "glucose": 0,
                "blood_pressure": 0,
                "skin": 0.5,  # Reset to minimum valid value
                "insulin": 0,
                "bmi": 0.0,
                "dpf": 0.0,
                "age": 0
            }
            # Clear prediction results
            if "prediction" in st.session_state:
                del st.session_state.prediction
                del st.session_state.patient
                del st.session_state.diabetes_prob
                del st.session_state.healthy_prob
            # Rerun to reflect changes
            st.rerun()
        
        if predict:
            # Validate all inputs
            errors = validate_required_fields(
                glucose, blood_pressure, bmi, age, dpf, skin, insulin, pregnancies
            )
            
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
                
                # Store current values for persistence
                st.session_state.form_values = {
                    "pregnancies": pregnancies,
                    "glucose": glucose,
                    "blood_pressure": blood_pressure,
                    "skin": skin,
                    "insulin": insulin,
                    "bmi": bmi,
                    "dpf": dpf,
                    "age": age
                }
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # =====================================================
    # CSV Upload
    # =====================================================
    if st.session_state.mode == "upload":
        uploaded_file = st.file_uploader(
            "📤 Upload CSV File",
            type=["csv","xlsx", "xls"],
            help="Upload a CSV file with the required columns"
        )
        
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
                
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
