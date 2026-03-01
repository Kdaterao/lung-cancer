# LungShield AI: Equitable Lung Cancer Risk Prediction

LungShield AI is a clinical decision support tool designed to identify high-risk individuals for lung cancer who are frequently missed by traditional industry standards. Built for the **BITS x IBM Datathon**, this project leverages machine learning to prioritize clinical sensitivity and health equity.

## The Problem: Underutilized Screening
* **235k** new lung cancer cases annually in the US.
* **130k** annual deaths, making it the leading cause of cancer death.
* **Only 16%** current national screening rate.
* **The Gap:** Current [USPSTF 2021 guidelines](https://docs.google.com/document/d/19VA1TndFpIRV3F__v2AdgNi4H-vE1tzzkq_DqDtqFwo/edit) rely on rigid age and smoking "pack-year" cutoffs, often ignoring environmental factors and disproportionately missing women and racial minorities.

## Key Results
* **98% Recall:** By adjusting our decision threshold to **0.15**, the model identifies nearly every at-risk individual in the test set.
* **Equity-First:** The model incorporates [occupational exposure and air pollution data](https://www.kaggle.com/datasets/aizahzeeshan/lung-cancer-risk-in-25-countries/data), allowing it to flag risks for non-smokers and vulnerable workers.
* **Enterprise Scalability:** Designed to be containerized and deployed on the [IBM LinuxONE Community Cloud](https://github.com/linuxone-community-cloud/jupyter-lab-ml).

## Methodology
### 1. Data Balancing (Under-sampling)
The [original Kaggle dataset](https://www.kaggle.com/datasets/aizahzeeshan/lung-cancer-risk-in-25-countries/data) of 220,000+ records was highly imbalanced (only ~4% cancer cases). We utilized **Clinical Under-sampling** to create a 50/50 training set, forcing the AI to learn the specific biological and environmental signatures of lung cancer rather than just guessing "healthy."

### 2. Random Forest Architecture
We utilized an ensemble **Random Forest** approach (100 decision trees) to capture complex, non-linear interactions between variables like age, family history, and passive smoke exposure.

### 3. Sensitivity Thresholding
Standard AI models use a 0.50 certainty threshold. LungShield AI uses a **0.15 threshold** because, in oncology, the cost of a missed diagnosis (False Negative) far outweighs the cost of an extra screening (False Positive).

## Technical Architecture
* **Model:** Random Forest Classifier (Scikit-Learn).
* **Deployment:** Streamlit Dashboard for real-time clinical assessment.
* **Infrastructure:** Scalable architecture optimized for [IBM LinuxONE](https://github.com/linuxone-community-cloud/jupyter-lab-ml) using the **BITS26** EventCode.

## Repository Structure
* `app.py`: The Streamlit front-end interface.
* `backend.py`: The model logic, encoding functions, and prediction engine.
* `Lung_Cancer_Prediction.ipynb`: The research and development notebook showing the full data analysis.
* `rf_model.pkl`: The trained Random Forest "brain."
* `model_columns.pkl`: The feature alignment map for deployment.

## Team: Group 08
* **Sai Chekka**
* **Krish Daterao**

---
*Developed for the BITS x IBM Datathon 2026.*
