from flask import Flask, request, jsonify, render_template, send_from_directory, session
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
from flask_cors import CORS
import requests
import logging
import os
import random

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("SUPABASE_URL or SUPABASE_KEY missing in environment.")
    raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY then restart")

# Initialize Supabase client with error handling
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
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
CORS(app, supports_credentials=True)

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

def generate_motivational_message(emotion):
    messages = {
        "joy": [
            "Your happiness is contagious! Keep spreading joy.",
            "Wonderful! Remember this feeling on cloudy days.",
            "Celebrate your joy—you deserve it!"
        ],
        "sadness": [
            "It's okay to feel sad. This feeling will pass.",
            "You are not alone. Better days are ahead.",
            "Be gentle with yourself. You're doing the best you can."
        ],
        "anger": [
            "Take a deep breath. You can handle this.",
            "Anger is a natural emotion. Use it as fuel for positive change.",
            "Try to step back and see the bigger picture."
        ],
        "fear": [
            "Courage is not the absence of fear, but acting in spite of it.",
            "You have overcome fears before. You can do it again.",
            "Take small steps. You are braver than you think."
        ],
        "disgust": [
            "Disgust can be a signal to change something. What can you learn?",
            "Use this feeling to motivate positive action.",
            "Sometimes, disgust protects us from harm. Listen to it."
        ],
        "surprise": [
            "Embrace the unexpected! It can lead to new opportunities.",
            "Life is full of surprises. Stay open to them.",
            "A surprise can be a gift in disguise."
        ],
        "neutral": [
            "Every day is a new beginning. What will you do today?",
            "Balance is key. Enjoy the calm moments.",
            "Neutral feelings are okay. They give you space to reflect."
        ]
    }
    return random.choice(messages.get(emotion, ["You are doing great. Keep it up!"]))

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
            
            # Check if user has premium status
            try:
                profile_response = supabase.table("profiles").select("premium").eq("user_id", auth_response.user.id).execute()
                if hasattr(profile_response, "data") and profile_response.data and len(profile_response.data) > 0:
                    session['user']['premium'] = profile_response.data[0]['premium']
                else:
                    # Create profile if it doesn't exist
                    supabase.table("profiles").insert({
                        "user_id": auth_response.user.id,
                        "premium": False
                    }).execute()
                    session['user']['premium'] = False
            except Exception as e:
                logging.error(f"Error checking premium status: {e}")
                session['user']['premium'] = False
                
            logging.info(f"User logged in: {auth_response.user.email}")
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
            
            # Create user profile
            try:
                supabase.table("profiles").insert({
                    "user_id": auth_response.user.id,
                    "premium": False
                }).execute()
                session['user']['premium'] = False
            except Exception as e:
                logging.error(f"Error creating profile: {e}")
                session['user']['premium'] = False
                
            logging.info(f"User signed up: {auth_response.user.email}")
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
    logging.info("User logged out")
    return jsonify({"message": "Logged out successfully"})

@app.route('/user')
def get_user():
    user = session.get('user')
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Not authenticated"}), 401

@app.route('/upgrade', methods=['POST'])
def upgrade_to_premium():
    # Check authentication
    auth_error = require_auth()
    if auth_error:
        return auth_error
        
    user_id = session['user']['id']
    
    try:
        # Update user's premium status
        response = supabase.table("profiles").update({"premium": True}).eq("user_id", user_id).execute()
        
        if hasattr(response, "error") and response.error:
            logging.error(f"Upgrade error: {response.error}")
            return jsonify({"error": "Upgrade failed"}), 500
            
        # Update session
        session['user']['premium'] = True
        
        return jsonify({"message": "Upgraded to premium successfully"})
    except Exception as e:
        logging.exception("Failed to upgrade to premium")
        return jsonify({"error": str(e)}), 500

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
    # Check authentication
    auth_error = require_auth()
    if auth_error:
        return auth_error
        
    user_id = session['user']['id']
    logging.info(f"User ID from session: {user_id}")
    
    payload = request.get_json(silent=True)
    if payload and isinstance(payload, dict):
        user_input = payload.get("text") or payload.get("journal_entry")
    else:
        user_input = request.form.get("journal_entry") or request.form.get("text")
    if not user_input or not user_input.strip():
        return jsonify({"error": "No input provided"}), 400
    user_input = user_input.strip()
    
    logging.info(f"User input: {user_input}")

    detected = None
    if HF_HEADERS:
        detected = detect_emotion_with_hf(user_input)
    if not detected:
        detected = fallback_detect(user_input)
    emotion_label, emotion_score, emotions = detected
    
    # Generate motivational message
    message = generate_motivational_message(emotion_label)
    
    logging.info(f"Detected emotion: {emotion_label}, score: {emotion_score}")

    try:
        data = {
            "user_id": user_id,
            "text": user_input, 
            "emotion": emotion_label, 
            "message": message
            # Removed created_at as it's automatically set by the database
        }
        
        logging.info(f"Attempting to insert data: {data}")
        
        insert_response = supabase.table("emotions").insert(data).execute()
        
        logging.info(f"Insert response: {insert_response}")
        
        if hasattr(insert_response, "error") and insert_response.error:
            logging.error(f"Supabase insert error: {insert_response.error}")
            return jsonify({"error": "Failed to insert to Supabase", "details": str(insert_response.error)}), 500
            
    except Exception as e:
        logging.exception("Supabase insert failed")
        return jsonify({"error": "Failed to insert to Supabase", "details": str(e)}), 500

    return jsonify({
        "emotion": emotion_label,
        "score": int(emotion_score * 100),
        "message": message,
        "all_emotions": emotions
    })

@app.route('/history')
def get_history():
    # Check authentication
    auth_error = require_auth()
    if auth_error:
        return auth_error
        
    user_id = session['user']['id']
    
    try:
        # Only fetch entries for the current user
        response = supabase.table("emotions").select("text, emotion, message, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute()
        if hasattr(response, "data"):
            return jsonify(response.data)
        return jsonify({"error": "No data returned from Supabase"}), 500
    except Exception as e:
        logging.exception("Failed to fetch history")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
