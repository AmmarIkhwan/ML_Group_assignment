# ML_Group_assignment
Group Assignment Machine Learning 
# 🎓 Engineering Graduate Salary Predictor | 🇮🇳

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://engineering-graduate-salary-predictor.streamlit.app/)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow.svg)](https://catboost.ai/)

An AI-powered web application that predicts engineering graduate salaries in India based on academic performance, technical skills, and personal attributes using advanced machine learning techniques.

## 🚀 Live Demo

**Try the app:** [https://engineering-graduate-salary-predictor.streamlit.app/](https://engineering-graduate-salary-predictor.streamlit.app/)

## 📊 Project Overview

This project leverages machine learning to predict starting salaries for engineering graduates in India. The model considers multiple factors including academic records, technical skills assessments, personality traits, and educational background to provide accurate salary estimates.

### 🎯 Key Features

- **Individual Predictions**: Interactive form for personalized salary predictions
- **Batch Processing**: Upload CSV files for multiple candidate predictions
- **Comprehensive Analysis**: Considers 29+ features including academics, skills, and personality
- **Realistic Estimates**: Provides salary ranges based on model uncertainty
- **User-Friendly Interface**: Clean, intuitive web application built with Streamlit

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **R² Score** | 0.2851 |
| **RMSE** | ₹117,051 |
| **MAE** | ₹89,310 |
| **MAPE** | 43.18% |

**Prediction Accuracy:**
- Within 10% of actual: 22.3%
- Within 20% of actual: 42.6%
- Within 30% of actual: 57.5%

**Best Model:** Tuned CatBoost Regressor (selected from 15+ algorithms)

## 📈 Dataset & Features

The model analyzes **29 key features** across multiple categories:

### 📚 Academic Performance
- 10th & 12th grade percentages and boards
- College GPA, tier, and specialization
- Graduation year and college location

### 🧠 Aptitude & Skills Assessment
- English, Logical, and Quantitative reasoning scores
- Domain-specific knowledge assessments
- Technical skills in 7 engineering areas

### 👤 Personal Attributes
- Big Five personality traits (conscientiousness, agreeableness, etc.)
- Demographic information

## 🛠️ Technical Implementation

### Data Processing Pipeline
1. **Data Cleaning**: Handles missing values and invalid entries (-1 conversions)
2. **Feature Engineering**: Creates composite scores and performance indicators
3. **Categorical Encoding**: Intelligent board classification and label encoding
4. **Feature Selection**: Optimized feature set for best performance

### Model Training & Selection
- **15+ Algorithms Tested**: Including LightGBM, Random Forest, Neural Networks, XGBoost
- **Hyperparameter Tuning**: Grid search optimization for top performers
- **Cross-Validation**: Robust evaluation methodology
- **Ensemble Methods**: Simple ensemble for improved accuracy

### Deployment Architecture
- **Frontend**: Streamlit web application
- **Backend**: Scikit-learn pipeline with CatBoost
- **Data Validation**: Comprehensive input validation and error handling
- **Scalability**: Supports both individual and batch predictions

## 📁 Project Structure

```
├── Model_Training/
│   └── A2.ipynb                          # Complete EDA and model training notebook
│   └── Engineering_graduate_salary.csv   # Dataset used for training
├── app8.py                               # Streamlit web application
├── final_salary_prediction_pipeline.pkl  # Trained model pipeline
├── model_metadata.pkl              # Model performance metrics
├── india.png                       # App logo/header image
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/engineering-salary-predictor.git
   cd engineering-salary-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app8.py
   ```

4. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Fill in the prediction form or upload a CSV file

### CSV Upload Format

For batch predictions, upload a CSV with these 29 columns:
```
Gender, DOB, 10percentage, 10board, 12percentage, 12board, CollegeTier, 
Degree, Specialization, collegeGPA, CollegeCityTier, CollegeState, 
GraduationYear, English, Logical, Quant, Domain, Computer
