import streamlit as st
import pandas as pd
import os

st.title("📊 Model Monitoring Dashboard")

LOG_FILE = "logs/monitoring_logs.csv"

# ✅ Check if log file exists
if not os.path.exists(LOG_FILE):
    st.warning("No monitoring logs found yet.")
    st.info("Run the prediction app and submit feedback to generate logs.")
    st.stop()

# Load logs
df = pd.read_csv(LOG_FILE)

st.subheader("Recent Predictions")
st.dataframe(df.tail(10))

# --- Average latency ---
st.subheader("Average Latency by Model")
latency = df.groupby("model_version")["latency_ms"].mean()
st.bar_chart(latency)

# --- Feedback score ---
st.subheader("Average Feedback Score")
feedback = df.groupby("model_version")["feedback_score"].mean()
st.bar_chart(feedback)

# --- Comments ---
st.subheader("Recent User Comments")
comments = df[df["comments"].notna()][["timestamp", "model_version", "comments"]]
st.dataframe(comments.tail(5))
