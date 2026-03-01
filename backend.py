import kagglehub
import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



#----- encode data ------

def encode_df(df):
    df = df.fillna("None")
    columns_to_drop = ["ID", "Country", "Early_Detection", "Treatment_Type", "Cancer_Stage", "Adenocarcinoma_Type", "Mortality_Rate", "Survival_Years"]
    cleaned = df.drop(columns=columns_to_drop)
    encoding_cols = [col for col in cleaned.columns if not is_numeric_dtype(df[col]) and len(cleaned[col].value_counts()) > 2]
    one_hot_encoded = pd.get_dummies(cleaned, columns=encoding_cols, drop_first=True)
    for col in one_hot_encoded.columns:
        if not is_numeric_dtype(one_hot_encoded[col]):
            labels, uniques = pd.factorize(one_hot_encoded[col])
            one_hot_encoded[col] = labels
        if is_bool_dtype(one_hot_encoded[col]):
            one_hot_encoded[col] = one_hot_encoded[col].astype(int)
    return one_hot_encoded


def trainer():
    path = kagglehub.dataset_download("aizahzeeshan/lung-cancer-risk-in-25-countries")
    df = pd.read_csv(path + "/lung_cancer_prediction_dataset.csv")
    df = df[df['Country'] == 'USA']

    processed_df = encode_df(df) #function defined above
    cancer_df = processed_df[processed_df['Lung_Cancer_Diagnosis'] == 1]
    healthy_df = processed_df[processed_df['Lung_Cancer_Diagnosis'] == 0]
    healthy_sample = healthy_df.sample(n=len(cancer_df), random_state=42)
    balanced_df = pd.concat([cancer_df, healthy_sample])

    X = balanced_df.drop(columns=["Lung_Cancer_Diagnosis", "Healthcare_Access"])
    y = balanced_df["Lung_Cancer_Diagnosis"]

    # Split & Train (Using Random Forest with Balanced Weights)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)

    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(list(X.columns), "model_columns.pkl")

    print("Model Trained Successfully.\n")

def predict(user_input, model, model_columns, threshold=0.15):
    """
    Predict lung cancer risk for a single patient.

    Parameters:
    - user_input: dict of patient features
    - model: trained classifier (RandomForest or similar)
    - model_columns: list of columns used in training
    - threshold: probability above which risk is considered 'High'

    Returns:
    - dict with prediction label, probability, and recommendation
    """

    # Convert user input into a DataFrame and align with training columns
    input_vector = pd.DataFrame([user_input])
    input_vector = input_vector.reindex(columns=model_columns, fill_value=0)

    # Predict probability
    pred_prob = model.predict_proba(input_vector)[0][1]  # Probability of Class 1

    # Determine risk label
    risk_label = "High" if pred_prob >= threshold else "Low"

    # Recommendation based on risk
    recommendation = (
        "REFER FOR LOW-DOSE CT SCREENING" if risk_label == "High"
        else "MONITOR & RE-EVALUATE ANNUALLY"
    )

    # Create human-readable result
    result = {
        "prediction": risk_label,
        "probability": round(pred_prob, 4),  # Probability as float (0–1)
        "recommendation": recommendation
    }

    return result


