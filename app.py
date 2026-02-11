# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)   # ✅ allow requests from React

# -------------------------------
# LOAD MODELS
# -------------------------------
category_model = joblib.load("category_model.pkl")
tfidf = joblib.load("tfidf.pkl")

RISK_MODEL_DIR = "risk_models"

# -------------------------------
# HELPER FUNCTION FOR PREDICTIONS
# -------------------------------
def get_ml_predictions(text):
    """Get category and risk predictions for text"""
    if not text or not text.strip():
        return {"category": "Unknown", "risk": "Unknown"}
    
    try:
        # CATEGORY PREDICTION
        category = category_model.predict([text])[0]
        safe_cat = category.replace(" ", "_").lower()

        # Check if risk model exists for this category
        kmeans_path = os.path.join(RISK_MODEL_DIR, f"{safe_cat}_kmeans.pkl")
        map_path = os.path.join(RISK_MODEL_DIR, f"{safe_cat}_risk_map.pkl")

        if not os.path.exists(kmeans_path):
            return {"category": category, "risk": "Unknown"}

        # LOAD RISK MODEL
        risk_model = joblib.load(kmeans_path)
        risk_mapping = joblib.load(map_path)

        text_vec = tfidf.transform([text])
        cluster = int(risk_model.predict(text_vec)[0])
        risk_label = risk_mapping.get(cluster, "Unknown")

        return {
            "category": category,
            "risk": risk_label,
            "risk_level": risk_label  # For compatibility
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"category": "Unknown", "risk": "Unknown"}

# -------------------------------
# PREDICTION ROUTE (for testing)
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    
    text = data.get("description") or data.get("text") or data.get("content")
    
    if not text:
        return jsonify({"error": "description is required"}), 400
    
    predictions = get_ml_predictions(text)
    return jsonify(predictions)

# -------------------------------
# DIRECT PREDICTION ROUTE (for manual testing)
# -------------------------------
@app.route("/predict-text", methods=["POST"])
def predict_text():
    """Direct prediction endpoint for testing"""
    data = request.get_json()
    text = data.get("text", "")
    
    predictions = get_ml_predictions(text)
    return jsonify(predictions)

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)