# predictive_app.py
import time
import joblib
import pandas as pd
import streamlit as st
from log_utils import log_prediction

st.set_page_config(page_title="Flood Prediction App", layout="centered")
st.title("🌧 Flood Potential Prediction (Model v1 vs v2)")

@st.cache_resource
def load_models():
    return (
        joblib.load("models/model_v1.pkl"),
        joblib.load("models/model_v2.pkl"),
    )

model_v1, model_v2 = load_models()

# Session state
if "pred_ready" not in st.session_state:
    st.session_state.pred_ready = False

st.sidebar.header("Rainfall Inputs")

rain_1h = st.sidebar.number_input("Rain (1h)", 0.0)
rain_3h = st.sidebar.number_input("Rain (3h sum)", 0.0)
rain_6h = st.sidebar.number_input("Rain (6h sum)", 0.0)
rain_12h = st.sidebar.number_input("Rain (12h sum)", 0.0)

input_df = pd.DataFrame([{
    "rain_1h": rain_1h,
    "rain_3h_sum": rain_3h,
    "rain_6h_sum": rain_6h,
    "rain_12h_sum": rain_12h,
}])

st.subheader("Input Summary")
st.dataframe(input_df)

if st.button("Run Prediction"):
    start = time.time()

    pred_v1 = model_v1.predict(input_df)[0]
    pred_v2 = model_v2.predict(input_df)[0]

    latency = round((time.time() - start) * 1000, 2)

    st.session_state.pred_ready = True
    st.session_state.pred_v1 = pred_v1
    st.session_state.pred_v2 = pred_v2
    st.session_state.latency = latency

if st.session_state.pred_ready:
    st.subheader("Predictions")
    st.write(f"Model v1 Prediction: **{st.session_state.pred_v1}**")
    st.write(f"Model v2 Prediction: **{st.session_state.pred_v2}**")
    st.write(f"Latency: {st.session_state.latency} ms")

    st.subheader("Feedback")
    score = st.slider("Usefulness (1–5)", 1, 5, 4)
    comment = st.text_area("Comments")

    if st.button("Submit Feedback"):
        log_prediction(
            "v1",
            input_df.to_dict(),
            st.session_state.pred_v1,
            st.session_state.latency,
            score,
            comment,
        )
        log_prediction(
            "v2",
            input_df.to_dict(),
            st.session_state.pred_v2,
            st.session_state.latency,
            score,
            comment,
        )
        st.success("✅ Feedback & predictions logged")
