from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

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
# Map dropdown/string values to numbers
mapping = {
    "wbc": {"0-2":0, "3-5":1, "6-10":2, "11-20":3, ">20":4},
    "rbc": {"0-2":0, "3-5":1, "6-10":2, "11-20":3, ">20":4},
    "protein": {"Negative":0, "Trace":1, "1+":2, "2+":3, "3+":4, "4+":5},
    "bacteria": {"None":0, "Few":1, "Moderate":2, "Many":3},
    # Add more mappings as needed
}

def preprocess_input(data):
    """Convert string inputs to numeric for model."""
    processed = {}
    for key, value in data.items():
        if key in mapping:
            processed[key] = mapping[key].get(value, 0)  # default to 0 if unknown
        else:
            try:
                processed[key] = float(value)  # numeric input (like age)
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
        return ["Seek medical consultation immediately", "Consider urine culture test", "Increase water intake"]
    elif risk == "moderate":
        return ["Monitor symptoms for 48 hours", "Increase fluid intake", "Consult doctor if symptoms worsen"]
    else:
        return ["Low risk detected", "Maintain hydration", "Monitor for any changes"]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "Urinalysis Prediction API running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_row = preprocess_input(data)

        prediction = model.predict(input_row)[0]  # "positive"/"negative"
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

if __name__ == "__main__":
    app.run(debug=True)
