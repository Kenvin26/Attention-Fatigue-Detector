import pandas as pd
import joblib

model = joblib.load("fatigue_model.pkl")

def compute_fatigue():
    try:
        face = pd.read_csv("data/fatigue_logs.csv")
        latest = face.iloc[-1]
        X_input = [[latest["blinks"], latest["yawns"]]]
        prediction = model.predict(X_input)[0]
        return 90 if prediction == 1 else 40
    except:
        return 50 