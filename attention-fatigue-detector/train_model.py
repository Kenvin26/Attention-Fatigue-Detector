import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/fatigue_logs.csv")

# Simple label generation rule
df["fatigue"] = ((df["blinks"] > 20) | (df["yawns"] > 3)).astype(int)

# Features + label
X = df[["blinks", "yawns"]]
y = df["fatigue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "fatigue_model.pkl")
print("✅ Model saved to fatigue_model.pkl") 