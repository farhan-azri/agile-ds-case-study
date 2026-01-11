# monitor_dashboard.py
import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("📊 Model Monitoring Dashboard")

if not os.path.exists(LOG_PATH):
    st.warning("No monitoring logs found yet. Run prediction app and submit feedback.")
    st.stop()

df = pd.read_csv(LOG_PATH)

st.subheader("Recent Logs")
st.dataframe(df.tail(10))

st.subheader("Average Latency by Model")
st.bar_chart(df.groupby("model_version")["latency_ms"].mean())

st.subheader("Average Feedback Score")
st.bar_chart(df.groupby("model_version")["feedback_score"].mean())

st.subheader("Recent Comments")
comments = df[df["feedback_text"].str.strip() != ""].tail(5)
st.dataframe(comments[["timestamp", "model_version", "feedback_text"]])
