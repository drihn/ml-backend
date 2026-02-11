# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)  # allow requests from React

# -------------------------------
# LOAD ML MODELS
# -------------------------------
try:
    category_model = joblib.load("category_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML models: {e}")
    category_model = None
    tfidf = None

RISK_MODEL_DIR = "risk_models"

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.environ.get("DB_HOST"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            database=os.environ.get("DB_NAME"),
            port=int(os.environ.get("DB_PORT")),
            connection_timeout=10
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
        if category_model is None or tfidf is None:
            return {"category": "Model not loaded", "risk": "Model not loaded"}

        # CATEGORY PREDICTION
        category = category_model.predict([text])[0]
        safe_cat = category.replace(" ", "_").lower()

        # RISK PREDICTION
        kmeans_path = os.path.join(RISK_MODEL_DIR, f"{safe_cat}_kmeans.pkl")
        map_path = os.path.join(RISK_MODEL_DIR, f"{safe_cat}_risk_map.pkl")

        if not os.path.exists(kmeans_path):
            return {"category": category, "risk": "Unknown"}

        risk_model = joblib.load(kmeans_path)
        risk_mapping = joblib.load(map_path)

        text_vec = tfidf.transform([text])
        cluster = int(risk_model.predict(text_vec)[0])
        risk_label = risk_mapping.get(cluster, "Unknown")

        return {"category": category, "risk": risk_label, "risk_level": risk_label}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"category": "Unknown", "risk": "Unknown"}

# -------------------------------
# ROUTES
# -------------------------------

@app.route('/')
def home():
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

# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    ml_status = "loaded" if category_model else "not loaded"
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Barangay ML Backend',
        'ml_models': ml_status,
        'timestamp': datetime.now().isoformat()
    })

# Test Database
@app.route('/api/test-db', methods=['GET'])
def test_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as user_count FROM users")
        user_result = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) as report_count FROM reports")
        report_result = cursor.fetchone()
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
        return jsonify({'success': False, 'error': str(e)}), 500

# User Login
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, first_name, email, role, status, created_at 
            FROM users 
            WHERE email=%s AND password=%s
        """, (email, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            return jsonify({'success': True, 'message': 'Login successful', 'user': user})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

# Get Reports
@app.route('/api/reports', methods=['GET'])
def get_reports():
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
        return jsonify({'success': True, 'count': len(reports), 'reports': reports})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Create Report
@app.route('/api/reports', methods=['POST'])
def create_report():
    data = request.json
    description = data.get('description', '')
    ml_prediction = get_ml_predictions(description)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (user_id, incident_type, description, location, priority, response_status)
            VALUES (%s, %s, %s, %s, %s, 'pending')
        """, (
            data['user_id'],
            ml_prediction['category'],
            description,
            data.get('location', ''),
            ml_prediction['risk']
        ))
        conn.commit()
        report_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Report created', 'report_id': report_id, 'ml_prediction': ml_prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ML Prediction Endpoints
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

@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()
    text = data.get("text", "")
    predictions = get_ml_predictions(text)
    return jsonify(predictions)

# Statistics
@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
        user_stats = cursor.fetchall()
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
        return jsonify({'success': False, 'error': str(e)}), 500
    
    # -------------------------------
# USER REGISTRATION (SIGNUP)
# -------------------------------
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    first_name = data.get('first_name') or data.get('full_name') or data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    # Validate input
    if not first_name or not email or not password:
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        # Insert new user (default: citizen, pending approval)
        cursor.execute("""
            INSERT INTO users (first_name, email, password, role, status)
            VALUES (%s, %s, %s, 'citizen', 'pending')
        """, (first_name, email, password))
        
        conn.commit()
        user_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Registration successful! Please wait for admin approval.',
            'user_id': user_id
        })
        
    except Exception as e:
        print(f"❌ Signup error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    # -------------------------------
# GET PENDING USERS (ADMIN ONLY)
# -------------------------------
@app.route('/api/pending-users', methods=['GET'])
def get_pending_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get all pending users
        cursor.execute("""
            SELECT id, first_name, email, role, status, created_at
            FROM users
            WHERE status = 'pending'
            ORDER BY created_at DESC
        """)
        
        pending_users = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'count': len(pending_users),
            'users': pending_users
        })
        
    except Exception as e:
        print(f"❌ Error fetching pending users: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -------------------------------
# APPROVE USER (ADMIN ONLY)
# -------------------------------
@app.route('/api/approve-user', methods=['POST'])
def approve_user():
    data = request.json
    user_id = data.get('userId')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET status = 'approve' 
            WHERE id = %s AND status = 'pending'
        """, (user_id,))
        
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        
        if affected > 0:
            return jsonify({'success': True, 'message': 'User approved successfully'})
        else:
            return jsonify({'success': False, 'error': 'User not found or already approved'}), 404
            
    except Exception as e:
        print(f"❌ Error approving user: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# -------------------------------
# REJECT USER (ADMIN ONLY)
# -------------------------------
@app.route('/api/reject-user', methods=['POST'])
def reject_user():
    data = request.json
    user_id = data.get('userId')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET status = 'reject' 
            WHERE id = %s AND status = 'pending'
        """, (user_id,))
        
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        
        if affected > 0:
            return jsonify({'success': True, 'message': 'User rejected successfully'})
        else:
            return jsonify({'success': False, 'error': 'User not found or already processed'}), 404
            
    except Exception as e:
        print(f"❌ Error rejecting user: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    # -------------------------------
# ADMIN GET ALL REPORTS
# -------------------------------
@app.route('/api/admin/reports', methods=['GET'])
def admin_get_reports():
    admin_id = request.args.get('admin_id')
    
    # Optional: Verify admin
    if not admin_id:
        return jsonify({'error': 'admin_id required'}), 400
    
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
        
        return jsonify(reports), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------------------------------
# ADMIN UPDATE REPORT STATUS
# -------------------------------
@app.route('/api/admin/reports/<int:report_id>/status', methods=['PUT'])
def admin_update_report_status(report_id):
    data = request.json
    admin_id = data.get('admin_id')
    response_status = data.get('response_status')
    admin_notes = data.get('admin_notes', '')
    
    # Optional: Verify admin
    if not admin_id:
        return jsonify({'error': 'admin_id required'}), 400
    
    valid_statuses = ['pending', 'in_progress', 'responded', 'resolved', 'rejected']
    if response_status not in valid_statuses:
        return jsonify({'error': f'Invalid status. Must be one of: {valid_statuses}'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE reports 
            SET response_status = %s, admin_notes = %s, responded_at = NOW()
            WHERE id = %s
        """, (response_status, admin_notes, report_id))
        
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        
        if affected > 0:
            return jsonify({'success': True, 'message': 'Report status updated'}), 200
        else:
            return jsonify({'error': 'Report not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

    # -------------------------------
# CITIZEN GET MY REPORTS
# -------------------------------
@app.route('/api/reports/my/<int:user_id>', methods=['GET'])
def get_my_reports(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, user_id, incident_type, description, location, 
                   response_status, admin_notes, responded_at, 
                   created_at, updated_at, priority
            FROM reports 
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        
        reports = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify(reports), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------------------------------
# CITIZEN UPDATE MY REPORT
# -------------------------------
@app.route('/api/reports/<int:report_id>', methods=['PUT'])
def update_my_report(report_id):
    data = request.json
    user_id = data.get('user_id')
    incident_type = data.get('incident_type')
    description = data.get('description')
    location = data.get('location')
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    if not description:
        return jsonify({'error': 'description required'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if report belongs to user
        cursor.execute("SELECT id FROM reports WHERE id = %s AND user_id = %s", (report_id, user_id))
        if not cursor.fetchone():
            return jsonify({'error': 'Report not found or unauthorized'}), 404
        
        # Update report
        cursor.execute("""
            UPDATE reports 
            SET incident_type = %s, description = %s, location = %s, updated_at = NOW()
            WHERE id = %s AND user_id = %s
        """, (incident_type, description, location, report_id, user_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Report updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------------------------------
# CITIZEN DELETE MY REPORT
# -------------------------------
@app.route('/api/reports/<int:report_id>', methods=['DELETE'])
def delete_my_report(report_id):
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if report belongs to user
        cursor.execute("SELECT id FROM reports WHERE id = %s AND user_id = %s", (report_id, user_id))
        if not cursor.fetchone():
            return jsonify({'error': 'Report not found or unauthorized'}), 404
        
        # Delete report
        cursor.execute("DELETE FROM reports WHERE id = %s AND user_id = %s", (report_id, user_id))
        
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        
        if affected > 0:
            return jsonify({'success': True, 'message': 'Report deleted successfully'}), 200
        else:
            return jsonify({'error': 'Report not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    # -------------------------------
# CHANGE PASSWORD
# -------------------------------
@app.route('/api/change-password', methods=['PUT'])
def change_password():
    data = request.json
    user_id = data.get('userId')
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
    
    if not user_id or not current_password or not new_password:
        return jsonify({'error': 'All fields are required'}), 400
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if user exists and current password matches
        cursor.execute("SELECT id, password FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Verify current password (assuming plain text for now)
        if user['password'] != current_password:
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Update to new password
        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (new_password, user_id))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Password updated successfully'}), 200
        
    except Exception as e:
        print(f"❌ Change password error: {e}")
        return jsonify({'error': str(e)}), 500

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
