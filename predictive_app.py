import streamlit as st
import joblib
import time
from log_utils import log_prediction

# Load trained models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

st.title("🌧️ Flood Potential Prediction App")

st.markdown("Enter recent rainfall measurements (in mm):")

# ✅ Inputs MUST match training features (order matters!)
rain_1h = st.number_input("Rainfall in last 1 hour (mm)", min_value=0.0, value=0.0)
rain_3h_sum = st.number_input("Rainfall in last 3 hours (mm)", min_value=0.0, value=0.0)
rain_6h_sum = st.number_input("Rainfall in last 6 hours (mm)", min_value=0.0, value=0.0)
rain_12h_sum = st.number_input("Rainfall in last 12 hours (mm)", min_value=0.0, value=0.0)

# ✅ Feature vector aligned with training
inputs = [[
    rain_1h,
    rain_3h_sum,
    rain_6h_sum,
    rain_12h_sum
]]

if st.button("Predict Flood Risk"):
    # --- Model v1 ---
    start = time.time()
    pred_v1 = model_v1.predict(inputs)[0]
    latency_v1 = round((time.time() - start) * 1000, 2)

    # --- Model v2 ---
    start = time.time()
    pred_v2 = model_v2.predict(inputs)[0]
    latency_v2 = round((time.time() - start) * 1000, 2)

    st.subheader("🔎 Prediction Results")
    st.write(f"**Model v1 (Logistic Regression)**: `{pred_v1}`  \nLatency: `{latency_v1} ms`")
    st.write(f"**Model v2 (Random Forest)**: `{pred_v2}`  \nLatency: `{latency_v2} ms`")

    st.divider()

    # --- Feedback ---
    feedback_score = st.slider("Prediction Quality (1 = Poor, 5 = Excellent)", 1, 5, 3)
    comments = st.text_area("Comments / Observations")

    if st.button("Submit Feedback"):
        log_prediction("v1", inputs, pred_v1, latency_v1, feedback_score, comments)
        log_prediction("v2", inputs, pred_v2, latency_v2, feedback_score, comments)
        st.success("✅ Feedback successfully logged")
