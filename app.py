# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)   # ✅ allow requests from React

@app.route('/')
def home():
    """Root endpoint - para hindi 404"""
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

# -------------------------------
# LOAD ML MODELS
# -------------------------------
try:
    category_model = joblib.load("category_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    print("✅ ML Models loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load ML models: {e}")
    category_model = None
    tfidf = None

RISK_MODEL_DIR = "risk_models"

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
def get_db_connection():
    """Connect to Railway MySQL database"""
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
def get_ml_predictions(text):
    """Get category and risk predictions for text"""
    if not text or not text.strip():
        return {"category": "Unknown", "risk": "Unknown"}
    
    try:
        # Check if models are loaded
        if category_model is None or tfidf is None:
            return {"category": "Model not loaded", "risk": "Model not loaded"}
        
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
            "risk_level": risk_label
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"category": "Unknown", "risk": "Unknown"}

# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    ml_status = "loaded" if category_model else "not loaded"
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Barangay ML Backend',
        'ml_models': ml_status,
        'timestamp': datetime.now().isoformat()
    })

# -------------------------------
# DATABASE TEST ENDPOINT
# -------------------------------
@app.route('/api/test-db', methods=['GET'])
def test_database():
    """Test database connection"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get counts
        cursor.execute("SELECT COUNT(*) as user_count FROM users")
        user_result = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) as report_count FROM reports")
        report_result = cursor.fetchone()
        
        # Get recent reports
        cursor.execute("""
            SELECT r.id, r.incident_type, r.description, r.created_at, u.first_name 
            FROM reports r 
            JOIN users u ON r.user_id = u.id 
            ORDER BY r.created_at DESC 
            LIMIT 5
        """)
        recent_reports = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'database': 'connected',
            'stats': {
                'total_users': user_result['user_count'],
                'total_reports': report_result['report_count']
            },
            'recent_reports': recent_reports
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# -------------------------------
# USER LOGIN
# -------------------------------
@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, first_name, email, role, status, created_at 
            FROM users 
            WHERE email = %s AND password = %s
        """, (email, password))
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': user
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid email or password'
            }), 401
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

# -------------------------------
# REPORTS API
# -------------------------------
@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all reports"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT r.*, u.first_name, u.email 
            FROM reports r 
            JOIN users u ON r.user_id = u.id 
            ORDER BY r.created_at DESC
        """)
        reports = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'count': len(reports),
            'reports': reports
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reports', methods=['POST'])
def create_report():
    """Create new report with ML prediction"""
    data = request.json
    
    # Get ML predictions
    description = data.get('description', '')
    ml_prediction = get_ml_predictions(description)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO reports 
            (user_id, incident_type, description, location, priority, response_status) 
            VALUES (%s, %s, %s, %s, %s, 'pending')
        """, (
            data['user_id'],
            ml_prediction['category'],  # Use ML predicted category
            description,
            data.get('location', ''),
            ml_prediction['risk']  # Use ML predicted risk as priority
        ))
        
        conn.commit()
        report_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Report created successfully',
            'report_id': report_id,
            'ml_prediction': ml_prediction
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# -------------------------------
# ML PREDICTION ENDPOINTS
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """ML prediction endpoint (standalone)"""
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
# STATISTICS
# -------------------------------
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # User stats
        cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
        user_stats = cursor.fetchall()
        
        # Report stats
        cursor.execute("SELECT incident_type, COUNT(*) as count FROM reports GROUP BY incident_type")
        category_stats = cursor.fetchall()
        
        cursor.execute("SELECT priority, COUNT(*) as count FROM reports GROUP BY priority")
        priority_stats = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'users_by_role': user_stats,
                'reports_by_category': category_stats,
                'reports_by_priority': priority_stats
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)