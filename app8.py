import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure page
st.set_page_config(
    page_title="Engineering Graduate Salary Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for headers and styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0078d4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .prediction-amount {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
# ========================================================================================
# CUSTOM TRANSFORMERS (MUST MATCH TRAINING SCRIPT)
# ========================================================================================

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.graduation_year_mode_ = None
        
    def fit(self, X, y=None):
        grad_years = X['GraduationYear'][X['GraduationYear'] != 0]
        self.graduation_year_mode_ = grad_years.mode()[0] if len(grad_years) > 0 else 2014
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed.loc[X_transformed['GraduationYear'] == 0, 'GraduationYear'] = self.graduation_year_mode_
        
        technical_cols = ['ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
                         'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
        
        for col in technical_cols:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].replace(-1, 0)
        
        return X_transformed

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year=2014):
        self.reference_year = reference_year
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        X_transformed['Academic_Performance_Score'] = (
            X_transformed['10percentage'] + 
            X_transformed['12percentage'] + 
            X_transformed['collegeGPA']
        ) / 3
        
        X_transformed['Aptitude_Total_Score'] = (
            X_transformed['English'] + 
            X_transformed['Logical'] + 
            X_transformed['Quant']
        )
        
        technical_cols = ['ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
                         'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
        available_tech_cols = [col for col in technical_cols if col in X_transformed.columns]
        X_transformed['Technical_Skills_Count'] = (X_transformed[available_tech_cols] > 0).sum(axis=1)
        
        X_transformed['Positive_Personality'] = (
            X_transformed['conscientiousness'] + 
            X_transformed['agreeableness'] + 
            X_transformed['extraversion'] + 
            X_transformed['openess_to_experience']
        )
        X_transformed['Emotional_Stability'] = -X_transformed['nueroticism']
        X_transformed['Years_Since_Graduation'] = self.reference_year - X_transformed['GraduationYear']
        
        return X_transformed

class BoardClassifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def classify_board(self, board_name):
        if pd.isna(board_name) or str(board_name).lower() == 'unknown':
            return 'Unknown'
        s = str(board_name).lower()
        
        if 'cbse' in s:
            return 'CBSE'
        if 'icse' in s or 'isc' in s:
            return 'ICSE'
        if any(keyword in s for keyword in ['state board', 'ssc', 'sslc', 'matric', 'stateboard']):
            return 'State Board'
        if 'rbse' in s or 'rajasthan' in s:
            return 'RBSE'
        if 'up board' in s or 'uttar pradesh' in s:
            return 'UP Board'
        if 'mp board' in s or 'mpbse' in s:
            return 'MP Board'
        if 'wbbse' in s or 'west bengal' in s:
            return 'WB Board'
        if 'kseeb' in s or 'karnataka' in s:
            return 'Karnataka Board'
        if 'tamil' in s or 'tn state board' in s:
            return 'TN Board'
        if 'gujarat' in s:
            return 'Gujarat Board'
        if 'bseb' in s or 'bihar' in s:
            return 'Bihar Board'
        if 'andhra' in s or 'apssc' in s:
            return 'AP Board'
        if 'kerala' in s:
            return 'Kerala Board'
        if 'maharashtra' in s:
            return 'Maharashtra Board'
        return 'Other'
    
    def transform(self, X):
        X_transformed = X.copy()
        
        if '10board' in X_transformed.columns:
            X_transformed['10board_group'] = X_transformed['10board'].apply(self.classify_board)
        if '12board' in X_transformed.columns:
            X_transformed['12board_group'] = X_transformed['12board'].apply(self.classify_board)
        
        return X_transformed

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders_ = {}
        self.categorical_features = ['Gender', '10board_group', '12board_group', 'Degree', 'Specialization', 'CollegeState']
        
    def fit(self, X, y=None):
        for col in self.categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X_col = X[col].fillna('Unknown').astype(str)
                le.fit(X_col)
                self.encoders_[col] = le
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.categorical_features:
            if col in X_transformed.columns and col in self.encoders_:
                X_col = X_transformed[col].fillna('Unknown').astype(str)
                le = self.encoders_[col]
                
                mask = X_col.isin(le.classes_)
                X_col[~mask] = le.classes_[0]
                
                X_transformed[col + '_encoded'] = le.transform(X_col)
                X_transformed = X_transformed.drop(columns=[col])
        return X_transformed

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, include_scaling=False):
        self.include_scaling = include_scaling
        self.feature_names_ = None
        self.scaler_ = None
        
    def fit(self, X, y=None):
        final_numeric_features = [
            '10percentage', '12percentage', 'collegeGPA', 'English', 'Logical', 'Quant', 'Domain',
            'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism', 'openess_to_experience',
            'CollegeTier', 'CollegeCityTier', 'Years_Since_Graduation',
            'Academic_Performance_Score', 'Aptitude_Total_Score', 'Technical_Skills_Count',
            'Positive_Personality', 'Emotional_Stability'
        ]
        
        final_technical_features = [
            'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
            'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg'
        ]
        
        final_categorical_features = [
            'Gender_encoded', '10board_group_encoded', '12board_group_encoded', 
            'Degree_encoded', 'Specialization_encoded', 'CollegeState_encoded'
        ]
        
        self.feature_names_ = (
            final_numeric_features + 
            final_technical_features + 
            final_categorical_features
        )
        
        if self.include_scaling:
            available_features = [col for col in self.feature_names_ if col in X.columns]
            X_selected = X[available_features].fillna(0)
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_selected)
        
        return self
    
    def transform(self, X):
        available_features = [col for col in self.feature_names_ if col in X.columns]
        X_selected = X[available_features].copy()
        X_selected = X_selected.fillna(0)
        
        if self.include_scaling and self.scaler_ is not None:
            X_selected = pd.DataFrame(
                self.scaler_.transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index
            )
        
        return X_selected

# Load model
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('final_salary_prediction_pipeline.pkl')
        try:
            metadata = joblib.load('model_metadata.pkl')
        except FileNotFoundError:
            metadata = None
        return pipeline, metadata
    except FileNotFoundError:
        st.error("Model file not found! Please upload the model files.")
        return None, None
#st.image("india.png") 
#st.markdown("<div style='text-align: center;'><img src='india.png' width='300'></div>", unsafe_allow_html=True)
import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_image_base64("india.png")
st.markdown(
    f"""
    <div style='width:300px; height:100px; margin:auto; display:flex; align-items:center; justify-content:center;'>
        <img src='data:image/png;base64,{img_base64}' style='max-width:100%; max-height:100%; object-fit:contain;' />
    </div>
    """,
    unsafe_allow_html=True
)
# Main App
st.markdown('<h1 class="main-header">🎓 Engineering Graduate Salary Predictor | 🇮🇳 </h1>', unsafe_allow_html=True)

pipeline, metadata = load_model()
if pipeline is None:
    st.stop()

if metadata:
    st.markdown(f'<p class="sub-header">Model: {metadata["model_name"]} | R² Score: {metadata["performance"]["r2_score"]:.4f}</p>', unsafe_allow_html=True)

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

with st.form("salary_prediction_form"):
    # PERSONAL INFO
    #st.subheader('',divider='blue')
    st.markdown("<h3 style='text-align: center;'>👤 Personal Information</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    gender = col1.selectbox("Gender", ['Male', 'Female'])
    Male='m'
    Female='f'
    dob = col2.date_input("Date of Birth", value=datetime.date(2000, 1, 1))

    # ACADEMIC PERFORMANCE
    st.subheader('',divider='gray')
    st.markdown("<h3 style='text-align: center;'>💼 Academic Performance</h3>", unsafe_allow_html=True)
    #st.subheader("💼 Academic Performance")
    col1, col2 = st.columns(2)
    tenth_percentage = col1.number_input("10th Grade Percentage (%)", 0.0, 100.0, 0.00, 0.1)
    tenth_board = col2.selectbox("10th Grade Board", ['CBSE', 'ICSE', 'State Board', 'RBSE', 'UP Board', 'MP Board', 'WB Board', 'Karnataka Board', 'TN Board', 'Gujarat Board', 'Bihar Board', 'AP Board', 'Kerala Board', 'Maharashtra Board', 'Other'], index=None)
    twelfth_percentage = col1.number_input("12th Grade Percentage (%)", 0.0, 100.0, 0.00, 0.1)
    twelfth_board = col2.selectbox("12th Grade Board", ['CBSE', 'ICSE', 'State Board', 'RBSE', 'UP Board', 'MP Board', 'WB Board', 'Karnataka Board', 'TN Board', 'Gujarat Board', 'Bihar Board', 'AP Board', 'Kerala Board', 'Maharashtra Board', 'Other'], index=None)
    college_gpa = col1.number_input("College GPA (%)", 0.0, 100.0, 0.00, 0.01)

    # COLLEGE INFORMATION
    st.subheader('',divider='gray')
    st.markdown("<h3 style='text-align: center;'>🏩 College Information</h3>", unsafe_allow_html=True)
    #st.subheader("🏩 College Information")
    col1, col2 = st.columns(2)
    college_tier = col1.selectbox("College Tier", [1, 2, 3], index=None)
    degree = col2.selectbox("Degree", ['B.Tech/B.E.', 'M.Tech./M.E.', 'MCA', 'M.Sc. (Tech.)'],index=None)
    specialization = col1.selectbox("Specialization",[
            'computer networking', 'information science', 'information & communication technology',
            'chemical engineering', 'industrial & production engineering', 'industrial engineering',
            'instrumentation and control engineering', 'computer engineering', 'telecommunication engineering',
            'civil engineering', 'metallurgical engineering', 'ceramic engineering',
            'industrial & management engineering', 'mechanical and automation', 'control and instrumentation engineering',
            'electronics & telecommunications', 'electronics & instrumentation eng', 'information technology',
            'mechanical engineering', 'biomedical engineering', 'electronics and communication engineering',
            'electronics engineering', 'electrical engineering', 'computer science & engineering',
            'computer science and technology', 'electronics and electrical engineering', 'information science engineering',
            'computer application', 'automobile/automotive engineering', 'biotechnology',
            'electronics and instrumentation engineering', 'electrical and power engineering', 'instrumentation engineering',
            'applied electronics and instrumentation', 'mechatronics', 'electronics and computer engineering',
            'embedded systems technology', 'aeronautical engineering', 'computer and communication engineering',
            'mechanical & production engineering', 'electronics', 'other'
        ],index=None)
    college_city_tier = col2.selectbox("College City Tier", [ 0, 1 ], index=None)
    college_state = col1.selectbox("College State", [
            'Karnataka', 'Maharashtra', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 
            'West Bengal', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Bihar',
            'Andhra Pradesh', 'Kerala', 'Haryana', 'Punjab', 'Odisha', 'Other'
        ] ,index=None)
    graduation_year = col2.number_input("Graduation Year", 2000, 2025, 2012, 1)

    # TEST SCORES
    st.subheader('',divider='gray')
    st.markdown("<h3 style='text-align: center;'>📊 Test Scores</h3>", unsafe_allow_html=True)
    #st.subheader("📊 Test Scores")
    col1, col2 = st.columns(2)
    english_score = col1.number_input("English Score", 0, 1000, 0, 1)
    logical_score = col2.number_input("Logical Reasoning Score", 0, 1000, 0, 1)
    quant_score = col1.number_input("Quantitative Score", 0, 1000, 0, 1)
    domain_score = col2.number_input("Domain Knowledge Score (0-1)", 0.0, 1.0, 0.0001, 0.1, format="%0.7f")

    # TECHNICAL SKILLS
    st.subheader('',divider='gray')
    st.markdown("<h3 style='text-align: center;'>💻 Technical Skills ('0' if not tested)</h3>", unsafe_allow_html=True)
    #st.subheader("💻 Technical Skills ('0' if not tested)")
    col1, col2 = st.columns(2)
    computer_programming = col1.number_input("Computer Programming", 0, 1000, 0, 1)
    electronics_semicon = col2.number_input("Electronics & Semiconductor", 0, 1000, 0, 1)
    computer_science = col1.number_input("Computer Science", 0, 1000, 0, 1)
    mechanical_engg = col2.number_input("Mechanical Engineering", 0, 1000, 0, 1)
    electrical_engg = col1.number_input("Electrical Engineering", 0, 1000, 0, 1)
    telecom_engg = col2.number_input("Telecom Engineering", 0, 1000, 0, 1)
    civil_engg = col1.number_input("Civil Engineering", 0, 1000, 0, 1)

    # PERSONALITY TRAITS
    #st.subheader("🧠 Personality Traits")
    st.subheader('',divider='gray')
    st.markdown("<h3 style='text-align: center;'>🧠 Personality Traits</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    conscientiousness = col1.number_input("Conscientiousness", -10.0, 10.0, 0.1000, 0.1, format="%0.4f")
    agreeableness = col2.number_input("Agreeableness", -10.0, 10.0, 0.1000, 0.1, format="%0.4f")
    extraversion = col1.number_input("Extraversion", -10.0, 10.0, 0.1000, 0.1, format="%0.4f")
    neuroticism = col2.number_input("Neuroticism", -10.0, 10.0, 0.1000, 0.1, format="%0.4f")
    openness = col1.number_input("Openness to Experience", -10.0, 10.0, 0.1000, 0.1, format="%0.4f")

    st.markdown("---")
    predict_clicked = st.form_submit_button("Predict Salary")

all_filled = all(
    field is not None and field != ''
    for field in [
    gender, dob,
    tenth_percentage, tenth_board,
    twelfth_percentage, twelfth_board,
    college_gpa, college_tier, degree, specialization,
    college_city_tier, college_state, graduation_year,
    english_score, logical_score, quant_score, domain_score,
    conscientiousness, agreeableness, extraversion, neuroticism, openness
])

if predict_clicked:
    if not all_filled:
        st.warning("Please fill out all fields before predicting.")
    else:
        input_data = {
            'Gender': gender,
            'DOB': str(dob),
            '10percentage': tenth_percentage,
            '10board': tenth_board.lower(),
            '12percentage': twelfth_percentage,
            '12board': twelfth_board.lower(),
            'CollegeTier': college_tier,
            'Degree': degree,
            'Specialization': specialization,
            'collegeGPA': college_gpa / 10.0,
            'CollegeCityTier': college_city_tier,
            'CollegeState': college_state,
            'GraduationYear': graduation_year,
            'English': english_score,
            'Logical': logical_score,
            'Quant': quant_score,
            'Domain': domain_score,
            'ComputerProgramming': computer_programming,
            'ElectronicsAndSemicon': electronics_semicon,
            'ComputerScience': computer_science,
            'MechanicalEngg': mechanical_engg,
            'ElectricalEngg': electrical_engg,
            'TelecomEngg': telecom_engg,
            'CivilEngg': civil_engg,
            'conscientiousness': conscientiousness,
            'agreeableness': agreeableness,
            'extraversion': extraversion,
            'nueroticism': neuroticism,
            'openess_to_experience': openness
        }

        df = pd.DataFrame([input_data])

        try:
            with st.spinner('Calculating salary prediction...'):
                prediction = pipeline.predict(df)[0]

            st.markdown(f"""
            <div class="prediction-box">
                <h2>Salary Prediction Results</h2>
                <div class="prediction-amount">₹{prediction:,.2f}</div>
                <p>Predicted Annual Salary</p>
                <p>Monthly: ₹{prediction/12:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
