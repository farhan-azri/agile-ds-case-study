import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Model Monitoring Dashboard")

df = pd.read_csv("logs/monitoring_logs.csv")

st.subheader("Recent Predictions")
st.dataframe(df.tail(10))

# Latency comparison
st.subheader("Average Latency by Model")
latency = df.groupby("model_version")["latency_ms"].mean()
st.bar_chart(latency)

# Feedback comparison
st.subheader("Average Feedback Score")
feedback = df.groupby("model_version")["feedback_score"].mean()
st.bar_chart(feedback)

# Recent comments
st.subheader("Recent User Comments")
comments = df[df["comments"].notna()][["timestamp", "model_version", "comments"]]
st.dataframe(comments.tail(5))
