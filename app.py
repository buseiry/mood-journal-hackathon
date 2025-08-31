from flask import Flask, request, jsonify, render_template, send_from_directory, session
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
from flask_cors import CORS
import requests
import logging
import os
import uuid

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# required env vars
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or os.urandom(24).hex()

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.error("SUPABASE_URL or SUPABASE_KEY missing in environment.")
    raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY then restart")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"} if HUGGING_FACE_API_KEY else None

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = FLASK_SECRET

IS_PROD = os.getenv("FLASK_ENV", "").lower() == "production"
app.config.update(
    SESSION_COOKIE_SECURE=IS_PROD,   # secure cookies in prod
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Allow your deployed origin and localhost for dev.
CORS_ORIGINS = [
    "https://moodlab.up.railway.app",
    "https://claritymind.up.railway.app",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]
CORS(app, supports_credentials=True, resources={r"/*": {"origins": CORS_ORIGINS}})

# SQL to show if table missing
SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS emotions (
  id BIGSERIAL PRIMARY KEY,
  text TEXT NOT NULL,
  emotion TEXT NOT NULL,
  emotion_label TEXT,
  emotion_score INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

def table_ok() -> bool:
    try:
        # quick read to confirm table is reachable
        r = supabase.table("emotions").select("text").limit(1).execute()
        return True
    except Exception as e:
        logging.warning("Table check failed: %s", e)
        return False

def detect_emotion_with_hf(text: str):
    try:
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=20)
        result = resp.json()
        if isinstance(result, dict) and "error" in result:
            logging.warning("HF model error: %s", result.get("error"))
            return None
        if not isinstance(result, list) or not result:
            logging.warning("HF returned unexpected format")
            return None
        emotions = result[0]
        top = max(emotions, key=lambda x: x.get("score", 0))
        return top.get("label"), float(top.get("score", 0)), emotions
    except Exception as e:
        logging.warning("HF request failed: %s", e)
        return None

def fallback_detect(text: str):
    t = text.lower()
    buckets = {
        "joy": ["happy", "joy", "glad", "delighted", "excited"],
        "sadness": ["sad", "depressed", "unhappy", "down"],
        "anger": ["angry", "mad", "furious", "annoyed", "frustrated"],
        "fear": ["afraid", "scared", "fear", "anxious", "worried"],
        "disgust": ["disgust", "gross", "nasty"],
        "surprise": ["surprised", "shocked"],
    }
    scores = {k: 0.0 for k in buckets}
    for label, keys in buckets.items():
        for kw in keys:
            if kw in t:
                scores[label] += 1.0
    total = sum(scores.values())
    if total == 0:
        emotions = [
            {"label": "neutral", "score": 0.9},
            {"label": "joy", "score": 0.05},
            {"label": "sadness", "score": 0.02},
            {"label": "anger", "score": 0.01},
            {"label": "fear", "score": 0.01},
            {"label": "disgust", "score": 0.01},
        ]
        return "neutral", 0.9, emotions
    normalized = {k: v / total for k, v in scores.items()}
    emotions = [{"label": k, "score": float(v)} for k, v in normalized.items()]
    emotions.sort(key=lambda x: x["score"], reverse=True)
    top = emotions[0]
    return top["label"], float(top["score"]), emotions

def extract_user_from_response(resp):
    # supabase auth call may return dict or object
    if not resp:
        return None
    if isinstance(resp, dict):
        return resp.get("user") or resp.get("data") or resp.get("session", {}).get("user")
    return getattr(resp, "user", None)

@app.post("/login")
def login():
    try:
        data = request.get_json(silent=True) or {}
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        auth_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        user_obj = extract_user_from_response(auth_response)
        if user_obj and getattr(user_obj, "id", None):
            session["user"] = {"id": str(user_obj.id), "email": getattr(user_obj, "email", email)}
            return jsonify({"message": "Login successful", "user": session["user"]})
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        logging.exception("Login failed")
        return jsonify({"error": "Login failed"}), 500

@app.post("/signup")
def signup():
    try:
        data = request.get_json(silent=True) or {}
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400
        auth_response = supabase.auth.sign_up({"email": email, "password": password})
        user_obj = extract_user_from_response(auth_response)
        if user_obj and getattr(user_obj, "id", None):
            session["user"] = {"id": str(user_obj.id), "email": getattr(user_obj, "email", email)}
            return jsonify({"message": "Signup successful", "user": session["user"]})
        return jsonify({"error": "Signup failed"}), 400
    except Exception as e:
        logging.exception("Signup failed")
        return jsonify({"error": "Signup failed"}), 500

@app.post("/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})

@app.get("/user")
def get_user():
    user = session.get("user")
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Not authenticated"}), 401

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/sw.js")
def service_worker():
    return send_from_directory(app.static_folder, "sw.js")

@app.post("/analyze")
def analyze():
    # require user session (frontend sends credentials: include)
    user = session.get("user")
    # For dev, allow anonymous via sticky uuid if no session
    if not user:
        anon_id = session.get("_anon_id")
        if not anon_id:
            anon_id = str(uuid.uuid4())
            session["_anon_id"] = anon_id
        user_id = anon_id
    else:
        user_id = user.get("id")

    payload = request.get_json(silent=True) or {}
    user_input = payload.get("text") or payload.get("journal_entry") or request.form.get("journal_entry") or request.form.get("text")
    if not user_input or not user_input.strip():
        return jsonify({"error": "No input provided"}), 400
    user_input = user_input.strip()

    detected = None
    if HF_HEADERS:
        detected = detect_emotion_with_hf(user_input)
    if not detected:
        detected = fallback_detect(user_input)
    emotion_label, emotion_score, emotions = detected

    if not table_ok():
        logging.error("Supabase table 'emotions' missing or unreachable.")
        return jsonify({
            "error": "Supabase table 'emotions' missing. Create it with SQL.",
            "create_table_sql": SQL_CREATE_TABLE.strip()
        }), 500

    try:
        data = {
            "text": user_input,
            "emotion": emotion_label,
            "emotion_label": emotion_label,
            "emotion_score": int(emotion_score * 100),
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

@app.get("/history")
def get_history():
    user = session.get("user")
    user_id = user.get("id") if user else None
    # allow anon dev history if no user
    if not user_id:
        user_id = session.get("_anon_id")

    if not table_ok():
        return jsonify({
            "error": "Supabase table 'emotions' missing. Create it with SQL.",
            "create_table_sql": SQL_CREATE_TABLE.strip()
        }), 500

    try:
        # fetch recent rows (no filtering by user because table has no user_id column)
        res = supabase.table("emotions") \
            .select("text, emotion, emotion_label, emotion_score, created_at") \
            .order("created_at", desc=True) \
            .limit(50) \
            .execute()
        return jsonify(getattr(res, "data", []))
    except Exception as e:
        logging.exception("Failed to fetch history")
        return jsonify({"error": str(e)}), 500

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(app.static_folder, "favicon.ico")
    if os.path.exists(path):
        return send_from_directory(app.static_folder, "favicon.ico")
    return "", 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=not IS_PROD)
