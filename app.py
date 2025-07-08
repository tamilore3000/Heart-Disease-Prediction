import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CACHING MODEL AND DATA ---
# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path):
    """Loads the trained machine learning model."""
    model = joblib.load(model_path)
    return model

# Cache the data loading
@st.cache_data
def load_data(data_path):
    """Loads the dataset."""
    data = pd.read_csv(data_path)
    return data

# --- LOAD RESOURCES ---
try:
    model = load_model('xgb_best_model.joblib')
    data = load_data('combined.csv')
except FileNotFoundError:
    st.error("Error: 'xgb_best_model.joblib' or 'combined.csv' not found.")
    st.info("Please make sure the model and data files are in the same directory as the app script.")
    st.stop()


# --- FEATURE MAPPING DICTIONARIES ---
# These mappings should be IDENTICAL to the ones used during model training
SEX_MAP = {'M': 1, 'F': 0}
CHEST_PAIN_MAP = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
RESTING_ECG_MAP = {'Normal': 0, 'ST': 1, 'LVH': 2}
EXERCISE_ANGINA_MAP = {'N': 0, 'Y': 1}
ST_SLOPE_MAP = {'Up': 0, 'Flat': 1, 'Down': 2}

# Dictionary for display text (more user-friendly)
CHEST_PAIN_DISPLAY = {
    'ATA': 'Atypical Angina',
    'NAP': 'Non-Anginal Pain',
    'ASY': 'Asymptomatic',
    'TA': 'Typical Angina'
}
FASTING_BS_DISPLAY = {0: 'False (<= 120 mg/dl)', 1: 'True (> 120 mg/dl)'}


# --- PREDICTION PAGE ---
def prediction_page():
    st.title("Heart Disease Prediction")
    st.markdown("Enter the patient's details below to predict the likelihood of heart disease.")
    st.markdown("---")

    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            'Age', 
            min_value=20, max_value=80, value=54,
            help="Patient's age in years."
        )
        sex_display = st.selectbox(
            'Sex', 
            options=['M', 'F'],
            help="Patient's gender (M: Male, F: Female)."
        )
        chest_pain_type_display = st.selectbox(
            'Chest Pain Type', 
            options=list(CHEST_PAIN_DISPLAY.keys()),
            format_func=lambda x: CHEST_PAIN_DISPLAY[x],
            help="Type of chest pain experienced."
        )

    with col2:
        resting_bp = st.slider(
            'Resting Blood Pressure', 
            min_value=80, max_value=200, value=130,
            help="Resting blood pressure in mm Hg."
        )
        cholesterol = st.slider(
            'Cholesterol', 
            min_value=0, max_value=610, value=225,
            help="Serum cholesterol in mg/dl. Note: 0 can indicate missing data in this dataset."
        )
        fasting_bs_display = st.selectbox(
            'Fasting Blood Sugar > 120 mg/dl', 
            options=[0, 1],
            format_func=lambda x: FASTING_BS_DISPLAY[x],
            help="Is the fasting blood sugar greater than 120 mg/dl?"
        )

    with col3:
        resting_ecg_display = st.selectbox(
            'Resting ECG', 
            options=list(RESTING_ECG_MAP.keys()),
            help="Resting electrocardiogram results."
        )
        max_hr = st.slider(
            'Maximum Heart Rate', 
            min_value=60, max_value=202, value=140,
            help="Maximum heart rate achieved during exercise."
        )
        exercise_angina_display = st.selectbox(
            'Exercise-Induced Angina', 
            options=['N', 'Y'],
            help="Did the patient experience angina during exercise? (Y: Yes, N: No)."
        )

    # Sliders for continuous values that need more space
    oldpeak = st.slider(
        'ST Depression (Oldpeak)', 
        min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest."
    )
    st_slope_display = st.selectbox(
        'ST Slope', 
        options=list(ST_SLOPE_MAP.keys()),
        help="The slope of the peak exercise ST segment."
    )

    st.markdown("---")

    # Prediction button
    if st.button('**Predict Heart Disease**', use_container_width=True):
        # Map user inputs to the numerical format required by the model
        sex = SEX_MAP[sex_display]
        chest_pain_type = CHEST_PAIN_MAP[chest_pain_type_display]
        fasting_bs = int(fasting_bs_display)
        resting_ecg = RESTING_ECG_MAP[resting_ecg_display]
        exercise_angina = EXERCISE_ANGINA_MAP[exercise_angina_display]
        st_slope = ST_SLOPE_MAP[st_slope_display]

        # Create the feature vector for prediction
        input_data = np.array([[
            age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
        ]])

        # Make the prediction and get probability
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display the result
        st.markdown("### Prediction Result")
        if prediction == 1:
            probability = prediction_proba[1] * 100
            st.error(f"**Result: High Risk of Heart Disease**", icon="💔")
            st.metric(
                label="Probability of Heart Disease",
                value=f"{probability:.2f}%"
            )
            st.progress(float(probability / 100))

        else:
            probability = prediction_proba[0] * 100
            st.success(f"**Result: Low Risk of Heart Disease**", icon="❤️")
            st.metric(
                label="Probability of Not Having Heart Disease",
                value=f"{probability:.2f}%"
            )
            st.progress(float(probability / 100))


        st.info("**Disclaimer:** This prediction is based on a machine learning model and is not a substitute for professional medical advice. Please consult a doctor for any health concerns.")


# --- ABOUT PAGE ---
def about_page():
    st.title("About This Project")
    st.markdown('''
    This application is a tool for predicting the presence of heart disease in patients based on various health parameters. By leveraging a powerful XGBoost machine learning model, it aims to assist medical professionals and inform individuals about potential risks.

    ### Dataset Used
    The model was trained on a combined dataset from multiple sources, containing the following features:

    - **Age:** Age of the patient [years]
    - **Sex:** Sex of the patient [M, F]
    - **ChestPainType:** [TA, ATA, NAP, ASY]
    - **RestingBP:** Resting blood pressure [mm Hg]
    - **Cholesterol:** Serum cholesterol [mm/dl]
    - **FastingBS:** Fasting blood sugar > 120 mg/dl [1: true; 0: false]
    - **RestingECG:** [Normal, ST, LVH]
    - **MaxHR:** Maximum heart rate achieved [60-202]
    - **ExerciseAngina:** Exercise-induced angina [Y, N]
    - **Oldpeak:** ST depression induced by exercise
    - **ST_Slope:** Slope of the peak exercise ST segment [Up, Flat, Down]
    - **HeartDisease:** Output class [1: heart disease, 0: Normal]

    ### The Model
    After evaluating several models, including Logistic Regression, Random Forest, and SVM, an **XGBoost (Extreme Gradient Boosting)** classifier was selected for its high accuracy and robustness in handling this type of data. The model has been fine-tuned to provide the most reliable predictions possible.

    ---
    *This app was created by Tamilore Olaogun in 2023.*
    ''')

# --- CONTACT PAGE ---
def contact_page():
    st.title("Contact Information")
    st.markdown('''
    Thank you for using the Heart Disease Prediction app! If you have any feedback, questions, or suggestions, please feel free to reach out.

    **Creator:** Tamilore Olaogun

    **Email:** [olaoguntamilore@gmail.com](mailto:olaoguntamilore@gmail.com)

    **GitHub:** [github.com/tamilore3000](https://github.com/tamilore3000)
    ''')


# --- MAIN APP LOGIC ---
def main():
    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            'Main Menu',
            ['Prediction', 'About', 'Contact'],
            icons=['heart-pulse', 'info-circle', 'person-rolodex'],
            menu_icon="cast",
            default_index=0
        )

    # Page routing
    if selected == 'Prediction':
        prediction_page()
    elif selected == 'About':
        about_page()
    elif selected == 'Contact':
        contact_page()

if __name__ == "__main__":
    main()