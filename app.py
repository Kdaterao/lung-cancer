# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import kagglehub
from backend import trainer, encode_df, predict  # Your backend functions

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Lung Cancer Risk Dashboard", layout="wide")
st.title("Lung Cancer Risk Prediction Dashboard")

# -----------------------------
# 1. Load / Train Model
# -----------------------------
MODEL_PATH = "rf_model.pkl"

if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
    st.success("Model loaded from disk.")
else:
    with st.spinner("Training model..."):
        trainer()  # This should save rf_model.pkl
    rf_model = joblib.load(MODEL_PATH)
    st.success("Model trained and loaded.")

# Load model columns
if not os.path.exists("model_columns.pkl"):
    dfpath = kagglehub.dataset_download("aizahzeeshan/lung-cancer-risk-in-25-countries")
    df = pd.read_csv(dfpath + "/lung_cancer_prediction_dataset.csv")
    mldf = encode_df(df)
    model_columns = mldf.drop(columns=['Lung_Cancer_Diagnosis']).columns
    joblib.dump(model_columns, "model_columns.pkl")
else:
    model_columns = joblib.load("model_columns.pkl")

## -----------------------------
# 2. Two-Column Layout (Polished Tabs)
# -----------------------------
col1, col2 = st.columns([1, 2])  # left narrower, right wider

with col1:
    st.markdown("### 🧾 Patient Information", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Smoking", "Family & Environment", "Country & Population"])

    with tab1:
        st.markdown("**Basic Info**")
        Age = st.slider("Age", 1, 100, 30, help="Patient age in years")
        Gender_input = st.selectbox("Gender", ["Male", "Female"], help="Patient gender")

    with tab2:
        st.markdown("**Smoking History**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            Smoker_input = st.selectbox("Smoker?", ["No", "Yes"], help="Has the patient smoked regularly?")
            Years_of_Smoking = st.slider("Years of Smoking", 0, 80, 0, help="Total years the patient has smoked")
        with col_s2:
            Cigarettes_per_Day = st.slider("Cigarettes per Day", 0, 50, 0, help="Average cigarettes per day")
            Passive_Smoker_input = st.selectbox("Passive Smoker?", ["No", "Yes"], help="Exposure to secondhand smoke?")

    with tab3:
        st.markdown("**Family & Environment**")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            Family_History_input = st.selectbox("Family History?", ["No", "Yes"], help="Any family history of lung cancer?")
            Occupational_Exposure_input = st.selectbox("Occupational Exposure?", ["No", "Yes"], help="Exposure to hazardous substances at work?")
        with col_f2:
            Indoor_Pollution_input = st.selectbox("Indoor Pollution?", ["No", "Yes"], help="Exposure to indoor pollution (cooking fuel, dust)?")
            Healthcare_Access = st.selectbox("Healthcare Access (0=Good,1=Poor)", [0, 1], help="Access to quality healthcare")

    with tab4:
        st.markdown("**Country & Population**")
        Population_Size = st.number_input("Population Size (millions)", value=50, help="Population of the country in millions")
        Annual_Lung_Cancer_Deaths = st.number_input("Annual Lung Cancer Deaths", value=500, help="Number of deaths per year in country")
        Lung_Cancer_Prevalence_Rate = st.number_input("Lung Cancer Prevalence Rate (%)", value=1.5, help="Percentage of population with lung cancer")
        Country_input = st.selectbox("Country", ["USA", "China", "Other"])
        Air_Pollution_Exposure_Low_input = st.selectbox("Air Pollution Exposure Low?", ["No", "Yes"], help="Is air pollution exposure considered low?")

    st.markdown("")
    calculate = st.button("Predict Risk", help="Click to calculate lung cancer risk")

    # Convert categorical inputs to numeric after tabs
    Gender = 0 if Gender_input == "Male" else 1
    Smoker = 0 if Smoker_input == "No" else 1
    Passive_Smoker = 0 if Passive_Smoker_input == "No" else 1
    Family_History = 0 if Family_History_input == "No" else 1
    Occupational_Exposure = 0 if Occupational_Exposure_input == "No" else 1
    Indoor_Pollution = 0 if Indoor_Pollution_input == "No" else 1
    Air_Pollution_Exposure_Low = 1 if Air_Pollution_Exposure_Low_input == "Yes" else 0
    Country_USA = 1 if Country_input == "USA" else 0
    Country_China = 1 if Country_input == "China" else 0

    user_input = {
        'Age': Age,
        'Gender': Gender,
        'Smoker': Smoker,
        'Years_of_Smoking': Years_of_Smoking,
        'Cigarettes_per_Day': Cigarettes_per_Day,
        'Passive_Smoker': Passive_Smoker,
        'Family_History': Family_History,
        'Occupational_Exposure': Occupational_Exposure,
        'Indoor_Pollution': Indoor_Pollution,
        'Healthcare_Access': Healthcare_Access,
        'Population_Size': Population_Size,
        'Annual_Lung_Cancer_Deaths': Annual_Lung_Cancer_Deaths,
        'Lung_Cancer_Prevalence_Rate': Lung_Cancer_Prevalence_Rate,
        'Country_USA': Country_USA,
        'Country_China': Country_China,
        'Air_Pollution_Exposure_Low': Air_Pollution_Exposure_Low
    }

with col2:
    st.markdown("### 🩺 Risk Assessment", unsafe_allow_html=True)

    if calculate:
        result = predict(user_input, rf_model, model_columns)
        risk_prob = result['probability'] * 100
        pred_label = "High" if risk_prob > 50 else "Low"
        risk_color = "#FF4B4B" if pred_label == "High" else "#4CAF50"

        # Styled risk circle
        st.markdown(
            f"""
            <div style="
                width: 180px;
                height: 180px;
                border-radius: 90px;
                border: 6px solid {risk_color};
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 28px;
                font-weight: bold;
                color: {risk_color};
                margin-left: auto;
                margin-right: auto;
            ">
                {pred_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Risk Probability:** {:.2f}%".format(risk_prob))
        if pred_label == "High":
            st.markdown(
                "⚠️ Recommendation: Refer for Low-Dose CT Screening immediately. "
                "Patient exceeds the risk threshold."
            )
        else:
            st.markdown("✅ Recommendation: Routine monitoring and lifestyle counseling.")

    else:
        st.markdown("Enter patient details and click **Predict Risk** to see results.")
