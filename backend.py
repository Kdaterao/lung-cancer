import kagglehub
import joblib
from pandas.api.types import is_numeric_dtype, is_bool_dtype
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#----- load in data ------

def load():
    # Download latest version
    path = kagglehub.dataset_download("aizahzeeshan/lung-cancer-risk-in-25-countries")

    print("Path to dataset files:", path)






#----- encode data ------

def encode_df(df):
  # fill missing values
  df = df.fillna("None")

  # drop irrelavent or redundant columns
  columns_to_drop = ["ID", "Early_Detection", "Treatment_Type", "Cancer_Stage", "Adenocarcinoma_Type", "Mortality_Rate", "Survival_Years"]
  cleaned = df.drop(columns=columns_to_drop)

  # determine which columns can be one-hot encoded
  encoding_cols = []
  for col in cleaned.columns:
    # only convert non-numeric, non-binary columns
    if not is_numeric_dtype(df[col]) and len(cleaned[col].value_counts()) > 2:
      encoding_cols.append(col)

  # perform one hot encoding
  one_hot_encoded = pd.get_dummies(cleaned, columns=encoding_cols, drop_first=True)

  # convert binary categorical values columns, not numeric columns
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


    mldf = encode_df(df) #function defined above
    X = mldf.drop(columns=['Lung_Cancer_Diagnosis'])
    y = mldf['Lung_Cancer_Diagnosis']

    # Split & Train (Using Random Forest with Balanced Weights)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)

    print("Model Trained Successfully.\n")

    joblib.dump(rf_model, "rf_model.pkl")



def predict(user_input, model, model_columns):


    # Convert user input into a DataFrame and align with training columns
    input_vector = pd.DataFrame([user_input])
    input_vector = input_vector.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    pred_class = model.predict(input_vector)[0]
    pred_prob = model.predict_proba(input_vector)[0][1]  # Probability of Class 1

    # Create human-readable result
    result = {
        "prediction": "LUNG CANCER DETECTED" if pred_class == 1 else "No Cancer Detected",
        "probability": round(pred_prob, 4)  # Probability as float (0–1)
    }

    return result



