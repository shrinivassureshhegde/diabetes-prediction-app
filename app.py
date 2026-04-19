import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "sample" not in st.session_state:
    st.session_state.sample = False

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Sidebar
menu = st.sidebar.selectbox("Menu", ["Home", "Prediction", "Insights"])

theme_choice = st.sidebar.selectbox("Select Theme", ["Dark", "Light"])
st.session_state.theme = theme_choice.lower()

# Theme CSS
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    div.stButton > button {
        border-radius: 10px;
        padding: 10px;
        background-color: #2E86C1;
        color: white;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1B4F72;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #F5F5F5;
        color: black;
    }

    div.stButton > button {
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        color: black;
        border: 1px solid #2E86C1;
    }

    div.stButton > button:hover {
        background-color: #2E86C1;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
color = "#58D68D" if st.session_state.theme == "dark" else "#2E86C1"

st.markdown(f"""
<h2 style='text-align: center; color: {color};'>
🩺 AI Diabetes Risk Analyzer
</h2>
""", unsafe_allow_html=True)

# ---------------- HOME ----------------
if menu == "Home":
    st.write("ML-based Diabetes Risk Prediction System")

# ---------------- PREDICTION ----------------
elif menu == "Prediction":

    st.subheader("Enter Patient Details")

    st.info("""
    Typical Healthy Ranges:
    - Glucose: 70–140
    - Blood Pressure: 60–90
    - BMI: 18–30
    - Age: 20–60
    """)

    # Sample button
    if st.button("Use Sample Data"):
        st.session_state.sample = True

    # Preset dropdown
    preset = st.selectbox("Choose Example Case", ["None", "Low Risk", "High Risk"])

    if preset == "Low Risk":
        preg, glucose, bp, skin, insulin, bmi, dpf, age = 1, 85, 66, 29, 80, 26.6, 0.35, 31
    elif preset == "High Risk":
        preg, glucose, bp, skin, insulin, bmi, dpf, age = 6, 148, 72, 35, 130, 33.6, 0.62, 50
    else:
        preg = 2 if st.session_state.sample else 0
        glucose = 120 if st.session_state.sample else 0
        bp = 70 if st.session_state.sample else 0
        skin = 20 if st.session_state.sample else 0
        insulin = 80 if st.session_state.sample else 0
        bmi = 25.0 if st.session_state.sample else 0.0
        dpf = 0.5 if st.session_state.sample else 0.0
        age = 30 if st.session_state.sample else 1

    # Inputs
    preg = st.number_input("Pregnancies", 0, 20, value=preg)
    glucose = st.number_input("Glucose", 0, 200, value=glucose)
    bp = st.number_input("Blood Pressure", 0, 150, value=bp)
    skin = st.number_input("Skin Thickness", 0, 100, value=skin)
    insulin = st.number_input("Insulin", 0, 900, value=insulin)
    bmi = st.number_input("BMI", 0.0, 70.0, value=bmi)
    dpf = st.number_input("DPF", 0.0, 3.0, value=dpf)
    age = st.number_input("Age", 1, 120, value=age)

    # Prediction
    if st.button("Predict"):
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        data = scaler.transform(data)

        proba = model.predict_proba(data)[0][1]

        st.write(f"Diabetes Probability: {proba*100:.2f}%")

        if proba < 0.3:
            st.success("Low Risk")
        elif proba < 0.7:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

# ---------------- INSIGHTS ----------------
elif menu == "Insights":

    df = pd.read_csv("diabetes.csv")

    st.subheader("Dataset")
    st.dataframe(df.head())

    st.subheader("Diabetes Distribution")
    st.bar_chart(df["Outcome"].value_counts())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Glucose vs Outcome")
    st.scatter_chart(df[["Glucose", "Outcome"]])

    st.subheader("BMI Distribution")
    st.line_chart(df["BMI"])

    st.subheader("Statistics")
    st.write(df.describe())

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    features = df.drop("Outcome", axis=1).columns

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    st.pyplot(fig)