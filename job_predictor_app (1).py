# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Load model and label encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set page config
st.set_page_config(page_title="Job Category Predictor", layout="wide", page_icon="💼")

# Custom CSS for dark theme look
st.markdown("""
    <style>
    .main { background-color: #111; color: white; }
    div[data-testid="stSidebar"] > div:first-child { background-color: #222; }
    </style>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("This application uses machine learning to predict job categories based on the skills you input.")
    st.markdown("**How to use:**\n\n1. Enter your skills in the text area\n2. Click 'Predict Job Category'\n3. View the predicted category and confidence scores")
    st.success("Model Ready")

# Main UI
st.title("Job Category Predictor")
st.subheader("Predict job categories based on your skills")

input_method = st.radio("Choose input method:", ["Text Area", "Comma Separated"], horizontal=True)

if input_method == "Text Area":
    user_input = st.text_area("Describe your skills and experience:")
else:
    user_input = st.text_input("Enter skills separated by commas:")

if st.button("Predict Job Category"):
    if user_input.strip() == "":
        st.warning("Please enter your skills or experience.")
    else:
        X_input = [user_input]
        probs = model.predict_proba(X_input)[0]
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.markdown("Predicted Category")
        st.success(pred_label)

        st.markdown("Confidence Scores")
        prob_df = pd.DataFrame({
            "Job Domain": label_encoder.inverse_transform(np.arange(len(probs))),
            "Confidence": probs * 100
        }).sort_values("Confidence", ascending=False).reset_index(drop=True)

        for idx, row in prob_df.iterrows():
            st.write(f"{row['Job Domain']}: {row['Confidence']:.1f}%")
            st.progress(min(int(row['Confidence']), 100))
