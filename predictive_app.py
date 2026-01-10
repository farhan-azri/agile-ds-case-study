import streamlit as st
import joblib
import time
import os
from log_utils import log_prediction

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Load models
model_v1 = joblib.load("models/model_v1.pkl")
model_v2 = joblib.load("models/model_v2.pkl")

st.title("🌧️ Flood Prediction Application")

# --- Inputs (match your dataset) ---
rain_1h = st.number_input("Rainfall (last 1 hour)", value=0.0)
rain_3h = st.number_input("Rainfall (last 3 hours)", value=0.0)
rain_6h = st.number_input("Rainfall (last 6 hours)", value=0.0)
rain_12h = st.number_input("Rainfall (last 12 hours)", value=0.0)

inputs = [[rain_1h, rain_3h, rain_6h, rain_12h]]

# --- Store predictions in session state ---
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# --- Step 1: Predict ---
if st.button("Predict Flood Risk"):
    start = time.time()
    pred_v1 = model_v1.predict(inputs)[0]
    latency_v1 = round((time.time() - start) * 1000, 2)

    start = time.time()
    pred_v2 = model_v2.predict(inputs)[0]
    latency_v2 = round((time.time() - start) * 1000, 2)

    st.session_state.predictions = {
        "inputs": inputs,
        "v1": (pred_v1, latency_v1),
        "v2": (pred_v2, latency_v2),
    }

# --- Step 2: Display results & feedback ---
if st.session_state.predictions:
    pred_v1, latency_v1 = st.session_state.predictions["v1"]
    pred_v2, latency_v2 = st.session_state.predictions["v2"]

    st.subheader("Prediction Results")
    st.write(f"Model v1 Prediction: **{pred_v1}** (Latency: {latency_v1} ms)")
    st.write(f"Model v2 Prediction: **{pred_v2}** (Latency: {latency_v2} ms)")

    feedback = st.slider("Feedback Score (1 = Poor, 5 = Excellent)", 1, 5, 3)
    comments = st.text_area("Comments / Observations")

    # --- Step 3: Submit Feedback ---
    if st.button("Submit Feedback"):
        log_prediction("v1", inputs, pred_v1, latency_v1, feedback, comments)
        log_prediction("v2", inputs, pred_v2, latency_v2, feedback, comments)

        st.success("✅ Feedback logged successfully")
        st.session_state.predictions = None
