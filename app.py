# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from datetime import datetime
import mysql.connector

# -------------------------------
# APP INIT
# -------------------------------
app = Flask(__name__)
CORS(app)  # allow requests from frontend

# -------------------------------
# LOAD ML MODELS
# -------------------------------
try:
    category_model = joblib.load("category_model.pkl")
    tfidf = joblib.load("tfidf.pkl")  # ✅ must be fitted
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load ML models: {e}")
    category_model = None
    tfidf = None

RISK_MODEL_DIR = "risk_models"  # folder where risk models are stored

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
def get_db_connection():
    """Connect to MySQL database"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'yamanote.proxy.rlwy.net'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'EoJhzIWGIkfIyHV0EnPBrzGTMYKMpGyB'),
            database=os.getenv('DB_NAME', 'railway'),
            port=int(os.getenv('DB_PORT', 55190))
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        raise

# -------------------------------
# HELPER FUNCTION FOR PREDICTIONS
# -------------------------------
def get_ml_predictions(text: str):
    """Return category and risk prediction for a given text"""
    if not text or not text.strip():
        return {"category": "Unknown", "risk": "Unknown"}

    try:
        if category_model is None or tfidf is None:
            return {"category": "Model not loaded", "risk": "Model not loaded"}

        # 1️⃣ Predict category
        category = category_model.predict([text])[0]
        safe_cat = category.replace(" ", "_").lower()

        # 2️⃣ Check if risk model exists
        kmeans_path = os.path.join(RISK_MODEL_DIR, f"{safe_cat}_kmeans.pkl")
        map_path = os.path.join(RISK_MODEL_DIR, f"{safe_cat}_risk_map.pkl")
        if not os.path.exists(kmeans_path) or not os.path.exists(map_path):
            return {"category": category, "risk": "Unknown", "risk_level": "Unknown"}

        # 3️⃣ Load risk model and mapping
        risk_model = joblib.load(kmeans_path)
        risk_mapping = joblib.load(map_path)

        # 4️⃣ Transform text and predict risk cluster
        text_vec = tfidf.transform([text])
        cluster = int(risk_model.predict(text_vec)[0])
        risk_label = risk_mapping.get(cluster, "Unknown")

        return {"category": category, "risk": risk_label, "risk_level": risk_label}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"category": "Unknown", "risk": "Unknown", "risk_level": "Unknown"}

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        'service': 'Smart Barangay API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'database': '/api/test-db',
            'login': '/api/login',
            'reports': '/api/reports',
            'stats': '/api/stats',
            'predict': '/predict'
        },
        'message': 'Welcome to Smart Barangay Backend!'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'ml_models': "loaded" if category_model else "not loaded",
        'timestamp': datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    """ML prediction endpoint"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    text = data.get("description") or data.get("text") or data.get("content")
    if not text:
        return jsonify({"error": "description is required"}), 400

    predictions = get_ml_predictions(text)
    return jsonify(predictions)

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
