# train_model_v2.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/dataset.csv")

FEATURES = [
    "rain_1h",
    "rain_3h_sum",
    "rain_6h_sum",
    "rain_12h_sum",
]

X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/model_v2.pkl")

print("🚀 Model v2 trained (Random Forest)")
