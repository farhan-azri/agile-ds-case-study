import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Features (exclude date + target)
FEATURES = [
    "rain_1h"
    # ,
    # "rain_3h_sum",
    # "rain_6h_sum",
    # "rain_12h_sum",
]

X = df[FEATURES]
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("✅ Model v1 Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model_v1.pkl")

print("✅ Model v1 trained and saved to models/model_v1.pkl")
