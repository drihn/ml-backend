# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)   # ✅ allow requests from React

# -------------------------------
# LOAD MODELS
# -------------------------------
category_model = joblib.load("category_model.pkl")
tfidf = joblib.load("tfidf.pkl")

RISK_MODEL_DIR = "risk_models"

# -------------------------------
# TEST ROUTE
# -------------------------------
@app.route("/")
def home():
    return "ML API is running..."

# -------------------------------
# PREDICTION ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    text = data.get("description") or data.get("text")

    if not text:
        return jsonify({"error": "description is required"}), 400

    # -----------------------
    # CATEGORY PREDICTION
    # -----------------------
    category = category_model.predict([text])[0]

    safe_cat = category.replace(" ", "_").lower()

    kmeans_path = os.path.join(
        RISK_MODEL_DIR,
        f"{safe_cat}_kmeans.pkl"
    )

    map_path = os.path.join(
        RISK_MODEL_DIR,
        f"{safe_cat}_risk_map.pkl"
    )

    # If no risk model found
    if not os.path.exists(kmeans_path):
        return jsonify({
            "category": category,
            "risk": "Unknown"
        })

    # -----------------------
    # LOAD RISK MODEL
    # -----------------------
    risk_model = joblib.load(kmeans_path)
    risk_mapping = joblib.load(map_path)

    text_vec = tfidf.transform([text])

    cluster = int(risk_model.predict(text_vec)[0])
    risk_label = risk_mapping.get(cluster, "Unknown")

    return jsonify({
        "category": category,
        "risk": risk_label
    })

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
