from flask import Flask, request, jsonify, render_template, send_from_directory, session
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
from flask_cors import CORS
import requests
import logging
import os

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Initialize Supabase client with error handling
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Test the connection
    supabase.table("emotions").select("id").limit(1).execute()
    logging.info("Supabase connection successful")
except Exception as e:
    logging.error(f"Failed to connect to Supabase: {e}")
    logging.error("Please check your SUPABASE_URL and SUPABASE_KEY environment variables")
    supabase = None

HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"} if HUGGING_FACE_API_KEY else None

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Configure session cookies for production
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)
CORS(app, supports_credentials=True, origins=["https://claritymind.up.railway.app"])

SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS emotions (
  id bigserial PRIMARY KEY,
  user_id UUID NOT NULL,
  text text NOT NULL,
  emotion text NOT NULL,
  created_at timestamptz DEFAULT now()
);
"""

def table_exists():
    try:
        r = supabase.table("emotions").select("user_id").limit(1).execute()
        if hasattr(r, "data") and r.data is not None:
            return True
        return False
    except Exception as e:
        logging.error("Error checking table: %s", e)
        return False

def detect_emotion_with_hf(text):
    try:
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=15)
        result = resp.json()
        if isinstance(result, dict) and "error" in result:
            logging.warning("HF model error: %s", result.get("error"))
            return None
        if not isinstance(result, list) or len(result) == 0:
            logging.warning("HF returned unexpected format")
            return None
        emotions = result[0]
        top = max(emotions, key=lambda x: x.get("score", 0))
        return top.get("label"), float(top.get("score", 0)), emotions
    except Exception as e:
        logging.warning("HF request failed: %s", e)
        return None

def fallback_detect(text):
    t = text.lower()
    buckets = {
        "joy": ["happy", "joy", "glad", "delighted", "excited"],
        "sadness": ["sad", "depressed", "unhappy", "down"],
        "anger": ["angry", "mad", "furious", "annoyed", "frustrated"],
        "fear": ["afraid", "scared", "fear", "anxious", "worried"],
        "disgust": ["disgust", "gross", "nasty"],
        "surprise": ["surprised", "shocked"],
    }
    scores = {k:0.0 for k in buckets}
    for label, keys in buckets.items():
        for kw in keys:
            if kw in t: scores[label] += 1.0
    total = sum(scores.values())
    if total == 0:
        emotions = [
            {"label": "neutral", "score": 0.9},
            {"label": "joy", "score": 0.025},
            {"label": "sadness", "score": 0.02},
            {"label": "anger", "score": 0.01},
            {"label": "fear", "score": 0.01},
            {"label": "disgust", "score": 0.01},
        ]
        return "neutral", 0.9, emotions
    normalized = {k: v / total for k,v in scores.items()}
    emotions = [{"label": k, "score": float(v)} for k,v in normalized.items()]
    emotions.sort(key=lambda x: x["score"], reverse=True)
    top = emotions[0]
    return top["label"], float(top["score"]), emotions

# User authentication routes
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        
        # Sign in with Supabase Auth
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if hasattr(auth_response, 'user') and auth_response.user:
            # Store user session
            session['user'] = {
                'id': str(auth_response.user.id),
                'email': auth_response.user.email
            }
            return jsonify({"message": "Login successful", "user": session['user']})
        else:
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        logging.error("Login error: %s", e)
        return jsonify({"error": "Login failed"}), 500

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        
        # Create user with Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if hasattr(auth_response, 'user') and auth_response.user:
            # Also log the user in after signup
            session['user'] = {
                'id': str(auth_response.user.id),
                'email': auth_response.user.email
            }
            return jsonify({
                "message": "Signup successful", 
                "user": session['user']
            })
        else:
            return jsonify({"error": "Signup failed"}), 400
            
    except Exception as e:
        logging.error("Signup error: %s", e)
        return jsonify({"error": "Signup failed"}), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})

@app.route('/user')
def get_user():
    user = session.get('user')
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Not authenticated"}), 401

# Middleware to check authentication
def require_auth():
    if 'user' not in session:
        return jsonify({"error": "Authentication required"}), 401
    return None

@app.route('/')
def home():
    return render_template("index.html")

# Serve service worker file from static via root path
@app.route('/sw.js')
def service_worker():
    return send_from_directory(app.static_folder, 'sw.js')

@app.route('/analyze', methods=['POST'])
def analyze():
    if supabase is None:
        return jsonify({"error": "Database not configured"}), 500
        
    # Check authentication
    auth_error = require_auth()
    if auth_error:
        return auth_error
        
    user_id = session['user']['id']
    
    payload = request.get_json(silent=True)
    if payload and isinstance(payload, dict):
        user_input = payload.get("text") or payload.get("journal_entry")
    else:
        user_input = request.form.get("journal_entry") or request.form.get("text")
    if not user_input or not user_input.strip():
        return jsonify({"error": "No input provided"}), 400
    user_input = user_input.strip()

    detected = None
    if HF_HEADERS:
        detected = detect_emotion_with_hf(user_input)
    if not detected:
        detected = fallback_detect(user_input)
    emotion_label, emotion_score, emotions = detected

    if not table_exists():
        logging.error("Supabase table 'emotions' missing. Provide SQL to create it.")
        return jsonify({
            "error": "Supabase table 'emotions' missing. Create it with SQL.",
            "create_table_sql": SQL_CREATE_TABLE.strip()
        }), 500

    try:
        data = {
            "user_id": user_id,
            "text": user_input, 
            "emotion": emotion_label, 
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        insert_response = supabase.table("emotions").insert(data).execute()
        if hasattr(insert_response, "error") and insert_response.error:
            logging.error("Supabase insert error: %s", insert_response.error)
    except Exception as e:
        logging.exception("Supabase insert failed")
        return jsonify({"error": "Failed to insert to Supabase", "details": str(e)}), 500

    return jsonify({
        "emotion": emotion_label,
        "score": int(emotion_score * 100),
        "message": "Mood recorded successfully",
        "all_emotions": emotions
    })

@app.route('/history')
def get_history():
    if supabase is None:
        return jsonify({"error": "Database not configured"}), 500
        
    # Check authentication
    auth_error = require_auth()
    if auth_error:
        return auth_error
        
    user_id = session['user']['id']
    
    if not table_exists():
        return jsonify({
            "error": "Supabase table 'emotions' missing. Create it with SQL.",
            "create_table_sql": SQL_CREATE_TABLE.strip()
        }), 500
    try:
        # Only fetch entries for the current user
        response = supabase.table("emotions").select("text, emotion, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
        if hasattr(response, "data"):
            return jsonify(response.data)
        return jsonify(response)
    except Exception as e:
        logging.exception("Failed to fetch history")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)