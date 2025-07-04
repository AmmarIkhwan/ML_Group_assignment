import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import base64
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io

# Configure page
st.set_page_config(
    page_title="Engineering Graduate Salary Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0078d4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-amount {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .uploadedFile {
        border: 2px dashed #0078d4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .feature-list {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0078d4;
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
        if pd.isna(board_name) or str(board_name).lower() == 'unknown' or str(board_name).strip() == '0':
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

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

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

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def display_header():
    """Display the main header with logo"""
    img_base64 = get_image_base64("india.png")
    if img_base64:
        st.markdown(
            f"""
            <div style='width:100%; height:150px; margin:auto; display:flex; align-items:center; justify-content:center; margin-bottom: 2rem;'>
                <img src='data:image/png;base64,{img_base64}' style='max-width:100%; max-height:100%; object-fit:contain;' />
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('<h1 class="main-header">🎓 Engineering Graduate Salary Predictor | 🇮🇳 </h1>', unsafe_allow_html=True)

def display_required_columns():
    """Display the required CSV columns format"""
    st.markdown("""
    <div class="info-box">
        <h3>📋 Required CSV Format</h3>
        <p>Your CSV file must contain exactly <strong>29 columns</strong> with the following headers:</p>
    </div>
    """, unsafe_allow_html=True)
    
    required_columns = [
        'Gender', 'DOB', '10percentage', '10board', '12percentage', '12board',
        'CollegeTier', 'Degree', 'Specialization', 'collegeGPA', 'CollegeCityTier',
        'CollegeState', 'GraduationYear', 'English', 'Logical', 'Quant', 'Domain',
        'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
        'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg',
        'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism',
        'openess_to_experience'
    ]
    
    # Display columns in a nice format
    col1, col2, col3 = st.columns(3)
    for i, column in enumerate(required_columns):
        if i % 3 == 0:
            col1.markdown(f"• **{column}**")
        elif i % 3 == 1:
            col2.markdown(f"• **{column}**")
        else:
            col3.markdown(f"• **{column}**")
    
    st.markdown("""
    <div class="feature-list">
        <h4>💡 Important Notes:</h4>
        <ul>
            <li><strong>Column Order</strong>: Columns don't need to be in the exact same position/order as shown above - just ensure all column names match exactly</li>
            <li><strong>collegeGPA</strong>: Should be on a scale of 0-10 (will be automatically normalized)</li>
            <li><strong>Board names</strong>: Use lowercase (e.g., 'cbse', 'icse', 'state board'). Use '0' or 'unknown' for unknown boards</li>
            <li><strong>Technical Skills</strong>: Use 0 if not tested, or -1 (will be automatically converted to 0)</li>
            <li><strong>Personality Traits</strong>: Typically range from -10 to 10</li>
            <li><strong>DOB</strong>: Use format YYYY-MM-DD</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ========================================================================================
# PAGE FUNCTIONS
# ========================================================================================

def individual_prediction_page():
    """Individual salary prediction form"""
    st.markdown("### 👤 Individual Salary Prediction")
    st.markdown("Fill out the form below to get a personalized salary prediction.")
    
    pipeline, metadata = load_model()
    if pipeline is None:
        st.stop()

    if metadata:
        rmse = metadata["performance"]["rmse"]
        r2 = metadata["performance"]["r2_score"]
        st.markdown(f'<p class="sub-header">Model: {metadata["model_name"]} | R² Score: {r2:.4f}</p>', unsafe_allow_html=True)

    with st.form("salary_prediction_form"):
        # PERSONAL INFO
        st.markdown("<h3 style='text-align: center; color: #0078d4;'>👤 Personal Information</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        gender = col1.selectbox("Gender", ['Male', 'Female'],index=None)
        dob = col2.date_input("Date of Birth", value=datetime.date(2000, 1, 1))

        # ACADEMIC PERFORMANCE
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #0078d4;'>📚 Academic Performance</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        tenth_percentage = col1.number_input("10th Grade Percentage (%)", 0.0, 100.0, 00.0, 0.1)
        tenth_board = col2.selectbox("10th Grade Board", ['CBSE', 'ICSE', 'State Board', 'RBSE', 'UP Board', 'MP Board', 'WB Board', 'Karnataka Board', 'TN Board', 'Gujarat Board', 'Bihar Board', 'AP Board', 'Kerala Board', 'Maharashtra Board', 'Other'],index=None)
        twelfth_percentage = col1.number_input("12th Grade Percentage (%)", 0.0, 100.0, 00.0, 0.1)
        twelfth_board = col2.selectbox("12th Grade Board", ['CBSE', 'ICSE', 'State Board', 'RBSE', 'UP Board', 'MP Board', 'WB Board', 'Karnataka Board', 'TN Board', 'Gujarat Board', 'Bihar Board', 'AP Board', 'Kerala Board', 'Maharashtra Board', 'Other'],index=None)
        college_gpa = col1.number_input("College GPA (%)", 0.0, 100.0, 00.0, 0.01)

        # COLLEGE INFORMATION
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #0078d4;'>🏫 College Information</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        college_tier = col1.selectbox("College Tier", [1, 2, 3],index=None)
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
        college_city_tier = col2.selectbox("College City Tier", [0, 1],index=None)
        college_state = col1.selectbox("College State", [
                'Karnataka', 'Maharashtra', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 
                'West Bengal', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Bihar',
                'Andhra Pradesh', 'Kerala', 'Haryana', 'Punjab', 'Odisha', 'Other'
            ],index=None)
        graduation_year = col2.number_input("Graduation Year", 2000, 2025, 2020, 1)

        # TEST SCORES
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #0078d4;'>📊 Test Scores</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        english_score = col1.number_input("English Score", 0, 1000, 500, 1)
        logical_score = col2.number_input("Logical Reasoning Score", 0, 1000, 500, 1)
        quant_score = col1.number_input("Quantitative Score", 0, 1000, 500, 1)
        domain_score = col2.number_input("Domain Knowledge Score (0-1)", 0.0, 1.0, 0.5, 0.01, format="%0.3f")

        # TECHNICAL SKILLS
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #0078d4;'>💻 Technical Skills ('0' if not tested)</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        computer_programming = col1.number_input("Computer Programming", 0, 1000, 0, 1)
        electronics_semicon = col2.number_input("Electronics & Semiconductor", 0, 1000, 0, 1)
        computer_science = col1.number_input("Computer Science", 0, 1000, 0, 1)
        mechanical_engg = col2.number_input("Mechanical Engineering", 0, 1000, 0, 1)
        electrical_engg = col1.number_input("Electrical Engineering", 0, 1000, 0, 1)
        telecom_engg = col2.number_input("Telecom Engineering", 0, 1000, 0, 1)
        civil_engg = col1.number_input("Civil Engineering", 0, 1000, 0, 1)

        # PERSONALITY TRAITS
        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: #0078d4;'>🧠 Personality Traits</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        conscientiousness = col1.number_input("Conscientiousness", -10.0, 10.0, 0.0, 0.1, format="%0.2f")
        agreeableness = col2.number_input("Agreeableness", -10.0, 10.0, 0.0, 0.1, format="%0.2f")
        extraversion = col1.number_input("Extraversion", -10.0, 10.0, 0.0, 0.1, format="%0.2f")
        neuroticism = col2.number_input("Neuroticism", -10.0, 10.0, 0.0, 0.1, format="%0.2f")
        openness = col1.number_input("Openness to Experience", -10.0, 10.0, 0.0, 0.1, format="%0.2f")

        st.markdown("---")
        predict_clicked = st.form_submit_button("🔮 Predict Salary", use_container_width=True, type="primary")

    if predict_clicked:
        # Validation: Check if all required fields are filled
        required_fields = {
            'Gender': gender,
            '10th Grade Board': tenth_board,
            '12th Grade Board': twelfth_board,
            'College Tier': college_tier,
            'Degree': degree,
            'Specialization': specialization,
            'College City Tier': college_city_tier,
            'College State': college_state
        }
        
        # Check for empty required fields
        empty_fields = [field_name for field_name, value in required_fields.items() if value is None]
        
        # Check for zero values in critical fields
        zero_fields = []
        if tenth_percentage == 0.0:
            zero_fields.append('10th Grade Percentage')
        if twelfth_percentage == 0.0:
            zero_fields.append('12th Grade Percentage')
        if college_gpa == 0.0:
            zero_fields.append('College GPA')
        if english_score == 0:
            zero_fields.append('English Score')
        if logical_score == 0:
            zero_fields.append('Logical Score')
        if quant_score == 0:
            zero_fields.append('Quantitative Score')
        if domain_score == 0.0:
            zero_fields.append('Domain Score')
        
        # Show validation errors
        if empty_fields or zero_fields:
            st.error("⚠️ Please complete all required fields before predicting:")
            if empty_fields:
                st.error(f"**Missing selections:** {', '.join(empty_fields)}")
            if zero_fields:
                st.warning(f"**Zero values detected:** {', '.join(zero_fields)} - Please enter valid scores")
            st.info("💡 **Tip:** All academic percentages and test scores should be greater than 0")
        else:
            # All validations passed, proceed with prediction
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
                with st.spinner('🔮 Calculating salary prediction...'):
                    prediction = pipeline.predict(df)[0]
                    
                    # Get RMSE for range calculation
                    rmse = metadata["performance"]["rmse"] if metadata else 50000  # fallback RMSE
                    lower_bound = max(0, prediction - rmse)  # Ensure non-negative
                    upper_bound = prediction + rmse

                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 Salary Prediction Results</h2>
                    <div class="prediction-amount">₹{prediction:,.0f}</div>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">Predicted Annual Salary</p>
                    <p style="font-size: 1.1rem; opacity: 0.9;">Monthly: ₹{prediction/12:,.0f}</p>
                    <p style="font-size: 0.9rem; opacity: 0.7; margin-top: 1.5rem;">Estimated Range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}</p>
                    <p style="font-size: 0.8rem; opacity: 0.7; margin-top: 1rem; font-style: italic;">Based on your profile and current market trends</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show disclaimer after prediction
                st.markdown("""
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                    <h4 style="color: #856404; margin-top: 0;">📝 Important Note</h4>
                    <p style="color: #856404; margin-bottom: 0; font-size: 0.9rem;">
                        This prediction is an estimate based on historical data and should be used as guidance only. 
                        Actual salaries may vary based on company, location, market conditions, and individual performance.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

def batch_prediction_page():
    """Batch prediction from CSV upload"""
    st.markdown("### 📊 Batch Salary Prediction")
    st.markdown("Upload a CSV file to predict salaries for multiple candidates at once.")
    
    pipeline, metadata = load_model()
    if pipeline is None:
        st.stop()

    if metadata:
        rmse = metadata["performance"]["rmse"]
        r2 = metadata["performance"]["r2_score"]
        st.markdown(f'<p class="sub-header">Model: {metadata["model_name"]} | R² Score: {r2:.4f}</p>', unsafe_allow_html=True)
    
    # Display required format
    display_required_columns()
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Upload Your CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the exact column format shown above"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            st.markdown("""
            <div class="success-box">
                <h4>✅ File Successfully Uploaded!</h4>
                <p>Preview of your data:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Rows", len(df))
            col2.metric("📋 Total Columns", len(df.columns))
            col3.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            required_columns = [
                'Gender', 'DOB', '10percentage', '10board', '12percentage', '12board',
                'CollegeTier', 'Degree', 'Specialization', 'collegeGPA', 'CollegeCityTier',
                'CollegeState', 'GraduationYear', 'English', 'Logical', 'Quant', 'Domain',
                'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience',
                'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg',
                'conscientiousness', 'agreeableness', 'extraversion', 'nueroticism',
                'openess_to_experience'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            extra_columns = [col for col in df.columns if col not in required_columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                return
            
            if extra_columns:
                st.warning(f"⚠️ Extra columns found (will be ignored): {', '.join(extra_columns)}")
            
            # Validation passed
            if len(missing_columns) == 0:
                st.success("✅ All required columns are present!")
                
                # Process collegeGPA if needed
                df_processed = df.copy()
                if 'collegeGPA' in df_processed.columns:
                    # Check if GPA values are > 10 (percentage scale) and convert to 0-10 scale
                    if df_processed['collegeGPA'].max() > 10:
                        df_processed['collegeGPA'] = df_processed['collegeGPA'] / 10.0
                        st.info("ℹ️ College GPA values converted from percentage to 0-10 scale")
                
                # Prediction button
                if st.button("🚀 Predict Salaries for All Candidates", type="primary", use_container_width=True):
                    try:
                        with st.spinner('🔮 Processing predictions for all candidates...'):
                            # Make predictions
                            predictions = pipeline.predict(df_processed[required_columns])
                            
                            # Get RMSE for range calculation
                            rmse = metadata["performance"]["rmse"] if metadata else 50000  # fallback RMSE
                            
                            # Add predictions to the original dataframe
                            results_df = df.copy()
                            results_df['Predicted_Salary'] = predictions
                            results_df['Predicted_Monthly_Salary'] = predictions / 12
                            results_df['Salary_Range_Lower'] = np.maximum(0, predictions - rmse)  # Ensure non-negative
                            results_df['Salary_Range_Upper'] = predictions + rmse
                            results_df['Prediction_Error_Range'] = f"±₹{rmse:,.0f}"
                            
                            # Display summary
                            st.markdown("""
                            <div class="prediction-box">
                                <h3>🎉 Batch Prediction Completed!</h3>
                                <p>Successfully predicted salaries for all candidates</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("📊 Total Predictions", len(predictions))
                            col2.metric("💰 Average Salary", f"₹{predictions.mean():,.0f}")
                            col3.metric("📈 Max Salary", f"₹{predictions.max():,.0f}")
                            col4.metric("📉 Min Salary", f"₹{predictions.min():,.0f}")
                            
                            # Show results preview
                            st.markdown("#### 📋 Results Preview")
                            display_columns = ['Gender', 'Degree', 'Specialization', 'GraduationYear', 
                                             'Predicted_Salary', 'Salary_Range_Lower', 'Salary_Range_Upper', 'Predicted_Monthly_Salary']
                            available_display_columns = [col for col in display_columns if col in results_df.columns]
                            
                            # Format the preview dataframe for better display
                            preview_df = results_df[available_display_columns].head(10).copy()
                            
                            # Round salary columns for cleaner display
                            salary_columns = ['Predicted_Salary', 'Salary_Range_Lower', 'Salary_Range_Upper', 'Predicted_Monthly_Salary']
                            for col in salary_columns:
                                if col in preview_df.columns:
                                    preview_df[col] = preview_df[col].round(0).astype(int)
                            
                            st.dataframe(preview_df, use_container_width=True)
                            
                            # Show disclaimer after batch results
                            st.markdown("""
                            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                                <h4 style="color: #856404; margin-top: 0;">📝 Important Note</h4>
                                <p style="color: #856404; margin-bottom: 0; font-size: 0.9rem;">
                                    These predictions are estimates based on historical data and should be used as guidance only. 
                                    Actual salaries may vary based on company, location, market conditions, and individual performance.
                                    Each prediction includes a salary range indicating potential variation.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Download link
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_data,
                                file_name=f"salary_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                type="primary"
                            )
                            
                    except Exception as e:
                        st.error(f"❌ Error during batch prediction: {str(e)}")
                        st.info("Please check your data format and try again.")
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")

# ========================================================================================
# MAIN APP
# ========================================================================================

def main():
    # Header
    display_header()
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h2 style='color: white; margin: 0;'>🧭 Navigation</h2>
        <p style='color: white; margin: 0; font-size: 0.9rem;'>Choose your prediction method</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Select Mode:",
        ["👤 Individual Prediction", "📊 Batch Prediction"],
        index=0
    )
    
    # Add some spacing and info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #0078d4;'>
        <h4 style='color: #0078d4; margin-top: 0;'>ℹ️ About</h4>
        <p style='font-size: 0.9rem; margin-bottom: 0;'>
            This AI-powered tool predicts engineering graduate salaries based on academic performance, 
            skills, and personal attributes using advanced machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "👤 Individual Prediction":
        individual_prediction_page()
    else:
        batch_prediction_page()

if __name__ == "__main__":
    main()
