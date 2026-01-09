import streamlit as st
import joblib
import time
from log_utils import log_prediction

# Load models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

st.title("🔮 Prediction Application")

# Example numeric inputs
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)

inputs = [[feature_1, feature_2, feature_3]]

if st.button("Predict"):
    # Model v1
    start = time.time()
    pred_v1 = model_v1.predict(inputs)[0]
    latency_v1 = round((time.time() - start) * 1000, 2)

    # Model v2
    start = time.time()
    pred_v2 = model_v2.predict(inputs)[0]
    latency_v2 = round((time.time() - start) * 1000, 2)

    st.subheader("Results")
    st.write(f"Model v1 Prediction: **{pred_v1}** (Latency: {latency_v1} ms)")
    st.write(f"Model v2 Prediction: **{pred_v2}** (Latency: {latency_v2} ms)")

    feedback = st.slider("Feedback Score (1–5)", 1, 5, 3)
    comments = st.text_area("Comments")

    if st.button("Submit Feedback"):
        log_prediction("v1", inputs, pred_v1, latency_v1, feedback, comments)
        log_prediction("v2", inputs, pred_v2, latency_v2, feedback, comments)
        st.success("✅ Feedback logged")
