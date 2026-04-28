import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Load model
diabetes_logistic_model = joblib.load(r"C:\Users\D.Anand\Desktop\diabetes_logistic_model.pkl")

# Example input
input_df = pd.DataFrame([[1,85,66,29,0,26.6,0.351,31]],
columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

# Prediction
prediction = diabetes_logistic_model.predict(input_df)

st.write("Prediction:", prediction)
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\D.Anand\Desktop\diabetes_logistic_model.pkl")

model = load_model()

st.title("Diabetes Prediction using Logistic Regression")
st.write("Enter patient details below to predict diabetes risk.")

pregnancies = st.number_input("Pregnancies", min_value=0.0, value=1.0, step=1.0)
glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
blood_pressure = st.number_input("BloodPressure", min_value=0.0, value=70.0)
skin_thickness = st.number_input("SkinThickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=79.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1.0, value=33.0, step=1.0)

input_df = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    model = pd.read_csv(r"C:\Users\D.Anand\Desktop\diabetes.csv")
    prob = diabetes_logistic_model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"Prediction: Diabetic (Probability: {prob:.2%})")
    else:
        st.success(f"Prediction: Non-Diabetic (Probability: {prob:.2%})")

st.markdown("### Deployment commands")
st.code("pip install streamlit scikit-learn pandas joblib")
st.code("streamlit run app.py")

# app.py

(r"C:\Users\D.Anand\Desktop\diabetes_logistic_model.pkl")
# diabetes_logistic_model.pkl
# diabetes.csv (optional for reference)

