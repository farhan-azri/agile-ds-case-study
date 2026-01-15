import time
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from log_utils import log_prediction

# ===============================
# STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(page_title="Flood Prediction App", layout="centered")
st.title("🌧 Flood Potential Prediction (Model Comparison)")
# ===============================

# FEATURES USED IN MODEL
# ===============================
FEATURES = ["rain_1h", "rain_3h_sum", "rain_6h_sum", "rain_12h_sum"]

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    return (
        joblib.load("models/model_v1.pkl"),
        joblib.load("models/model_v2.pkl"),
    )

model_v1, model_v2 = load_models()

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_dataset():
    df = pd.read_csv("data/dataset.csv")

    # Ensure date exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
    else:
        st.error("❌ dataset.csv must contain a 'date' or 'datetime' column.")
        st.stop()

    # Sort for clean charts
    df = df.sort_values("date").reset_index(drop=True)
    return df

df = load_dataset()

# ===============================
# SESSION STATE
# ===============================
if "pred_ready" not in st.session_state:
    st.session_state.pred_ready = False

# ===============================
# DATASET VISUALIZATION SECTION
# ===============================
st.subheader("📊 Dataset Overview")


with st.expander("Summary Statistics"):
    st.write(df[FEATURES].describe())


with st.expander("Rainfall Distributions"):
    # st.subheader("📈 Rainfall Trend (All Dates in Dataset)")

    chart_df = df.copy()
    chart_df = chart_df.groupby("date")[FEATURES].mean().reset_index()

    # Convert wide → long format for Plotly multi-line chart
    chart_long = chart_df.melt(
        id_vars="date",
        value_vars=FEATURES,
        var_name="Feature",
        value_name="Value"
    )

    fig = px.line(
        chart_long,
        x="date",
        y="Value",
        color="Feature",
        markers=False,
        # title="Average Rainfall Features Across All Dates"
    )

    fig.update_traces(mode="lines")
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        xaxis_tickformat="%Y-%m-%d"   # ✅ YYYY-MM-DD
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Hover on the chart to see all rainfall feature values for the selected date.")


# ===============================
# DATE SELECTION FOR AUTO INPUTS
# ===============================

available_dates = sorted(df["date"].dt.date.unique())
selected_date = st.sidebar.selectbox("Choose a date", available_dates)

# Take mean values from dataset (instead of latest row)
mean_values = df[FEATURES].describe().loc["mean"]

default_rain_1h = float(mean_values["rain_1h"])
default_rain_3h = float(mean_values["rain_3h_sum"])
default_rain_6h = float(mean_values["rain_6h_sum"])
default_rain_12h = float(mean_values["rain_12h_sum"])


st.sidebar.subheader("🌧 Rainfall Inputs (Mean)")
rain_1h = st.sidebar.number_input("Rain (1h) in mm", value=default_rain_1h)
rain_3h = st.sidebar.number_input("Rain (3h sum) in mm", value=default_rain_3h)
rain_6h = st.sidebar.number_input("Rain (6h sum) in mm", value=default_rain_6h)
rain_12h = st.sidebar.number_input("Rain (12h sum) in mm", value=default_rain_12h)

# Input dataframe for prediction
input_df_v1 = pd.DataFrame([{
    "rain_1h": rain_1h
}])

input_df_v2 = pd.DataFrame([{
    "rain_1h": rain_1h,
    "rain_3h_sum": rain_3h,
    "rain_6h_sum": rain_6h,
    "rain_12h_sum": rain_12h,
}])

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🚀 Run Prediction"):
    start = time.time()

    pred_v1 = model_v1.predict(input_df_v1)[0]
    pred_v2 = model_v2.predict(input_df_v2)[0]

    latency = round((time.time() - start) * 1000, 2)

    st.session_state.pred_ready = True
    st.session_state.pred_v1 = int(pred_v1)
    st.session_state.pred_v2 = int(pred_v2)
    st.session_state.latency = latency



# ===============================
# DISPLAY RESULTS + FEEDBACK
# ===============================
if st.session_state.pred_ready:
    st.subheader("✅ Predictions")

    st.write(f"📌 **Model v1 Prediction:** `{st.session_state.pred_v1}`")
    st.write(f"📌 **Model v2 Prediction:** `{st.session_state.pred_v2}`")
    st.write(f"⏱ **Latency:** {st.session_state.latency} ms")

    def interpret(pred):
        return "⚠️ Flood Risk HIGH" if pred == 1 else "✅ Flood Risk LOW"

    st.write("### Interpretation")
    st.write(f"Model v1: {interpret(st.session_state.pred_v1)}")
    st.write(f"Model v2: {interpret(st.session_state.pred_v2)}")

# ===============================
# FEEDBACK PER MODEL
# ===============================

# ===============================
# FEEDBACK PER MODEL
# ===============================
st.subheader("📝 Feedback (Per Model Version)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔹 Model v1")
    score_v1 = st.slider("Score (1–5)", 1, 5, 4, key="score_v1")
    comment_v1 = st.text_area("Comments", key="comment_v1")

with col2:
    st.markdown("#### 🔸 Model v2")
    score_v2 = st.slider("Score (1–5)", 1, 5, 4, key="score_v2")
    comment_v2 = st.text_area("Comments", key="comment_v2")


if st.button("📩 Submit Feedback"):
    # Log Model v1 feedback
    log_prediction(
        "v1",
        {"date": str(selected_date), **input_df_v1.iloc[0].to_dict()},
        st.session_state.pred_v1,
        st.session_state.latency,
        score_v1,
        comment_v1,
    )

    # Log Model v2 feedback
    log_prediction(
        "v2",
        {"date": str(selected_date), **input_df_v2.iloc[0].to_dict()},
        st.session_state.pred_v2,
        st.session_state.latency,
        score_v2,
        comment_v2,
    )

# if st.button("📩 Submit Feedback"):
#     # Log Model v1 feedback (full rainfall features)
#     log_prediction(
#         "v1",
#         {"date": str(selected_date), **input_df.iloc[0].to_dict()},
#         st.session_state.pred_v1,
#         st.session_state.latency,
#         score_v1,
#         comment_v1,
#     )

#     # Log Model v2 feedback (ONLY rain_1h)
#     log_prediction(
#         "v2",
#         {
#             "date": str(selected_date),
#             "rain_1h": float(input_df.iloc[0]["rain_1h"]),
#         },
#         st.session_state.pred_v2,
#         st.session_state.latency,
#         score_v2,
#         comment_v2,
#     )

    st.success("✅ Feedback logged separately for Model v1 and Model v2")


# st.subheader("📝 Feedback (Per Model Version)")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("#### 🔹 Feedback for Model v1")
#     score_v1 = st.slider("Usefulness v1 (1–5)", 1, 5, 4, key="score_v1")
#     comment_v1 = st.text_area("Comments for Model v1", key="comment_v1")

# with col2:
#     st.markdown("#### 🔸 Feedback for Model v2")
#     score_v2 = st.slider("Usefulness v2 (1–5)", 1, 5, 4, key="score_v2")
#     comment_v2 = st.text_area("Comments for Model v2", key="comment_v2")

# if st.button("📩 Submit Feedback"):
#     # Log Model v1 feedback
#     log_prediction(
#         "v1",
#         {"date": str(selected_date), **input_df.iloc[0].to_dict()},
#         st.session_state.pred_v1,
#         st.session_state.latency,
#         score_v1,
#         comment_v1,
#     )

#     # Log Model v2 feedback
#     log_prediction(
#         "v2",
#         {"date": str(selected_date), **input_df.iloc[0].to_dict()},
#         st.session_state.pred_v2,
#         st.session_state.latency,
#         score_v2,
#         comment_v2,
#     )

#     st.success("✅ Feedback logged separately for Model v1 and Model v2")
