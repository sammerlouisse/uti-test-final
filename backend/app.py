from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Input mapping for categorical features
# -----------------------------
mapping = {
    "wbc": {"0-2": 0, "3-5": 1, "6-10": 2, "11-20": 3, ">20": 4},
    "rbc": {"0-2": 0, "3-5": 1, "6-10": 2, "11-20": 3, ">20": 4},
    "protein": {"Negative": 0, "Trace": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5},
    "bacteria": {"None": 0, "Few": 1, "Moderate": 2, "Many": 3},
}

def preprocess_input(data):
    """Convert string inputs to numeric for model."""
    processed = {}
    for key, value in data.items():
        if key in mapping:
            processed[key] = mapping[key].get(value, 0)
        else:
            try:
                processed[key] = float(value)
            except:
                processed[key] = 0
    return [list(processed.values())]

# -----------------------------
# Risk & Recommendations
# -----------------------------
def determine_risk(proba: float) -> str:
    if proba >= 0.80:
        return "high"
    elif proba >= 0.55:
        return "moderate"
    else:
        return "low"

def get_recommendations(diagnosis: str, risk: str):
    if diagnosis.lower() == "negative":
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
# Routes
# -----------------------------
@app.route("/")
def home():
    return "Urinalysis Prediction API is running successfully 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_row = preprocess_input(data)

        prediction = model.predict(input_row)[0]
        proba = model.predict_proba(input_row)[0][1]
        confidence = round(proba * 100, 2)

        risk = determine_risk(proba)
        recommendations = get_recommendations(prediction, risk)

        return jsonify({
            "diagnosis": prediction,
            "riskLevel": risk,
            "confidence": confidence,
            "recommendations": recommendations
        })

    except Exception as e:
        print("Prediction ERROR:", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Production server for Render
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
