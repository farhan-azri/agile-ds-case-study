# log_utils.py
import os
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "monitoring_logs.csv")

os.makedirs(LOG_DIR, exist_ok=True)

def log_prediction(
    model_version,
    input_features,
    prediction,
    latency_ms,
    feedback_score,
    feedback_text,
):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "inputs": str(input_features),
        "prediction": int(prediction),
        "latency_ms": latency_ms,
        "feedback_score": feedback_score,
        "feedback_text": feedback_text or "",
    }

    df_new = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_new.to_csv(LOG_PATH, index=False)
    else:
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
