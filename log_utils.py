import csv
import os
from datetime import datetime

LOG_FILE = "logs/monitoring_logs.csv"

def log_prediction(model_version, inputs, prediction, latency, feedback_score, comments):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "model_version",
                "inputs",
                "prediction",
                "latency_ms",
                "feedback_score",
                "comments"
            ])

        writer.writerow([
            datetime.now(),
            model_version,
            str(inputs),
            prediction,
            latency,
            feedback_score,
            comments
        ])

