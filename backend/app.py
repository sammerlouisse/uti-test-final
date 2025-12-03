from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# RISK & RECOMMENDATIONS
# -----------------------------
def determine_risk(proba):
    if proba >= 0.80:
        return "high"
    elif proba >= 0.55:
        return "moderate"
    else:
        return "low"

def get_recommendations(diagnosis, risk):
    if diagnosis == "negative":
        return ["Maintain hydration", "Monitor symptoms", "Regular checkups"]

    if risk == "high":
        return [
            "Seek medical consultation immediately",
            "Consider urine culture test",
            "Increase water intake"
        ]
    elif risk == "moderate":
        return [
            "Monitor symptoms for 48 hours",
            "Increase fluid intake",
            "Consult doctor if symptoms worsen"
        ]
    else:
        return [
            "Low risk detected",
            "Maintain hydration",
            "Monitor for any changes"
        ]

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert input to 2D row
    input_row = [list(data.values())]

    # Predict
    prediction = model.predict(input_row)[0]
    proba = model.predict_proba(input_row)[0][1]  # probability of "positive"
    confidence = round(float(proba * 100), 2)

    risk = determine_risk(proba)
    recommendations = get_recommendations(prediction, risk)

    return jsonify({
        "diagnosis": prediction,
        "riskLevel": risk,
        "confidence": confidence,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
