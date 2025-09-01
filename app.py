from flask import Flask, request, jsonify, render_template, send_from_directory, session
from supabase import create_client, Client
from dotenv import load_dotenv
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
    logging.error("SUPABASE_URL or SUPABASE_KEY missing in environment")
    raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY then restart")

# Auth client (uses anon key). Use REST calls with the user's JWT so RLS will apply.
try:
    base_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logging.info("Supabase base client ready")
except Exception as e:
    logging.error("Failed to init Supabase client: %s", e)
    base_client = None

REST_URL = SUPABASE_URL.rstrip("/") + "/rest/v1"
HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"} if HUGGING_FACE_API_KEY else None

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# --- HTTP helpers for PostgREST (use user JWT in Authorization header) ---
def _rest_headers(token=None, prefer_return=False):
    headers = {
        "apikey": SUPABASE_KEY,
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers["Authorization"] = f"Bearer {SUPABASE_KEY}"
    if prefer_return:
        headers["Prefer"] = "return=representation"
    return headers

def rest_post(table, data, token=None, prefer_return=True, timeout=15):
    url = f"{REST_URL}/{table}"
    headers = _rest_headers(token=token, prefer_return=prefer_return)
    return requests.post(url, json=data, headers=headers, timeout=timeout)

def rest_get(table, params=None, token=None, timeout=15):
    url = f"{REST_URL}/{table}"
    headers = _rest_headers(token=token)
    return requests.get(url, params=params, headers=headers, timeout=timeout)

def rest_patch(table, params=None, data=None, token=None, prefer_return=True, timeout=15):
    url = f"{REST_URL}/{table}"
    headers = _rest_headers(token=token, prefer_return=prefer_return)
    return requests.patch(url, params=params, json=data, headers=headers, timeout=timeout)

# --- supabase-py response helpers ---
def extract_access_token(auth_resp):
    try:
        sess = getattr(auth_resp, "session", None)
        if sess:
            token = getattr(sess, "access_token", None)
            if token:
                return token
        if isinstance(auth_resp, dict):
            s = auth_resp.get("session") or (auth_resp.get("data", {}) or {}).get("session")
            if isinstance(s, dict):
                return s.get("access_token")
    except Exception:
        pass
    return None

def extract_user_obj(auth_resp):
    user = getattr(auth_resp, "user", None)
    if user:
        return user
    if isinstance(auth_resp, dict):
        u = auth_resp.get("user") or (auth_resp.get("data", {}) or {}).get("user")
        return u
    return None

def get_id_from_user(user_obj):
    if user_obj is None:
        return None
    if isinstance(user_obj, dict):
        return user_obj.get("id")
    return getattr(user_obj, "id", None)

def get_email_from_user(user_obj):
    if user_obj is None:
        return None
    if isinstance(user_obj, dict):
        return user_obj.get("email")
    return getattr(user_obj, "email", None)

# --- Emotion detection ---
def detect_emotion_with_hf(text):
    if not HF_HEADERS:
        return None
    try:
        resp = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            logging.warning("HF model error: %s", data.get("error"))
            return None
        if not isinstance(data, list) or len(data) == 0:
            logging.warning("HF returned unexpected format")
            return None
        emotions = data[0]
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
    scores = {k: 0.0 for k in buckets}
    for label, keys in buckets.items():
        for kw in keys:
            if kw in t:
                scores[label] += 1.0
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
    normalized = {k: v / total for k, v in scores.items()}
    emotions = [{"label": k, "score": float(v)} for k, v in normalized.items()]
    emotions.sort(key=lambda x: x["score"], reverse=True)
    top = emotions[0]
    return top["label"], float(top["score"]), emotions

def generate_motivational_message(emotion):
    messages = {
        "joy": ["Keep this joy. Share it.", "Wonderful. Hold this feeling."],
        "sadness": ["It is okay to feel sad.", "Be gentle with yourself."],
        "anger": ["Pause and breathe.", "Use this energy wisely."],
        "fear": ["Take a small step forward.", "You have faced fear before."],
        "disgust": ["Notice what to change.", "Use it to protect yourself."],
        "surprise": ["Stay open to new chances.", "A surprise can teach you."],
        "neutral": ["Small wins matter.", "Use calm to plan your next step."],
    }
    return random.choice(messages.get(emotion, ["You are doing fine."]))

# --- Auth check ---
def require_auth():
    if "user" not in session:
        return jsonify({"error": "Authentication required"}), 401
    return None

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/sw.js", methods=["GET"])
def service_worker():
    return send_from_directory(app.static_folder, "sw.js")

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        auth_resp = base_client.auth.sign_in_with_password({"email": email, "password": password})
        token = extract_access_token(auth_resp)
        user_obj = extract_user_obj(auth_resp)

        if not token or not user_obj:
            # Try to surface error from auth_resp if present
            err = None
            if isinstance(auth_resp, dict):
                err = auth_resp.get("error") or auth_resp.get("message")
            return jsonify({"error": "Invalid credentials", "details": err}), 401

        uid = get_id_from_user(user_obj)
        email_val = get_email_from_user(user_obj)

        session["user"] = {"id": str(uid), "email": email_val, "access_token": token}

        # Ensure profile exists and read premium flag
        try:
            params = {"select": "premium", "user_id": f"eq.{uid}"}
            r = rest_get("profiles", params=params, token=token)
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 0:
                session["user"]["premium"] = bool(r.json()[0].get("premium", False))
            else:
                r2 = rest_post("profiles", {"user_id": uid, "premium": False}, token=token)
                if r2.status_code in (200, 201):
                    session["user"]["premium"] = False
                else:
                    session["user"]["premium"] = False
                    logging.warning("Profile create returned %s %s", r2.status_code, r2.text)
        except Exception as e:
            logging.error("Profile check/create failed: %s", e)
            session["user"]["premium"] = False

        logging.info("User logged in: %s", session["user"]["email"])
        return jsonify({"message": "Login successful", "user": session["user"]})
    except Exception as e:
        logging.exception("Login error")
        return jsonify({"error": "Login failed", "details": str(e)}), 500

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        auth_resp = base_client.auth.sign_up({"email": email, "password": password})
        user_obj = extract_user_obj(auth_resp)
        if not user_obj:
            # signup failed, try to show server message
            err = None
            if isinstance(auth_resp, dict):
                err = auth_resp.get("error") or auth_resp.get("message")
            return jsonify({"error": "Signup failed", "details": err}), 400

        # Sign in to obtain token
        si = base_client.auth.sign_in_with_password({"email": email, "password": password})
        token = extract_access_token(si)
        uid = get_id_from_user(user_obj)
        email_val = get_email_from_user(user_obj)

        session["user"] = {"id": str(uid), "email": email_val, "access_token": token}

        # Create profile if needed
        try:
            params = {"select": "id", "user_id": f"eq.{uid}"}
            rcheck = rest_get("profiles", params=params, token=token)
            if rcheck.status_code == 200 and isinstance(rcheck.json(), list) and len(rcheck.json()) > 0:
                session["user"]["premium"] = False if not rcheck.json()[0].get("premium") else True
            else:
                r = rest_post("profiles", {"user_id": uid, "premium": False}, token=token)
                if r.status_code in (200, 201):
                    session["user"]["premium"] = False
                else:
                    session["user"]["premium"] = False
                    logging.warning("Profile insert returned %s %s", r.status_code, r.text)
        except Exception as e:
            logging.error("Error creating profile: %s", e)
            session["user"]["premium"] = False

        logging.info("User signed up: %s", session["user"]["email"])
        return jsonify({"message": "Signup successful", "user": session["user"]})
    except Exception as e:
        logging.exception("Signup error")
        return jsonify({"error": "Signup failed", "details": str(e)}), 500

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    logging.info("User logged out")
    return jsonify({"message": "Logged out successfully"})

@app.route("/user", methods=["GET"])
def get_user():
    user = session.get("user")
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Not authenticated"}), 401

@app.route("/upgrade", methods=["POST"])
def upgrade_to_premium():
    auth_err = require_auth()
    if auth_err:
        return auth_err
    uid = session["user"]["id"]
    token = session["user"].get("access_token")
    try:
        params = {"user_id": f"eq.{uid}"}
        r = rest_patch("profiles", params=params, data={"premium": True}, token=token)
        if r.status_code in (200, 204):
            session["user"]["premium"] = True
            return jsonify({"message": "Upgraded to premium successfully"})
        logging.error("Upgrade failed: %s %s", r.status_code, r.text)
        return jsonify({"error": "Upgrade failed", "details": r.text}), 500
    except Exception as e:
        logging.exception("Failed to upgrade to premium")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    auth_err = require_auth()
    if auth_err:
        return auth_err
    uid = session["user"]["id"]
    token = session["user"].get("access_token")
    payload = request.get_json(silent=True)
    if payload and isinstance(payload, dict):
        user_input = payload.get("text") or payload.get("journal_entry")
    else:
        user_input = request.form.get("journal_entry") or request.form.get("text")
    if not user_input or not user_input.strip():
        return jsonify({"error": "No input provided"}), 400
    user_input = user_input.strip()

    logging.info("User input: %s", user_input)
    detected = detect_emotion_with_hf(user_input) or fallback_detect(user_input)
    emotion_label, emotion_score, emotions = detected
    message = generate_motivational_message(emotion_label)
    logging.info("Detected emotion: %s score: %s", emotion_label, emotion_score)

    try:
        data = {"user_id": uid, "text": user_input, "emotion": emotion_label, "message": message}
        r = rest_post("emotions", data, token=token)
        if r.status_code not in (200, 201):
            logging.error("Supabase insert failed: %s %s", r.status_code, r.text)
            return jsonify({"error": "Failed to insert to Supabase", "details": r.text}), 500
    except Exception as e:
        logging.exception("Supabase insert failed")
        return jsonify({"error": "Failed to insert to Supabase", "details": str(e)}), 500

    return jsonify({"emotion": emotion_label, "score": int(emotion_score * 100), "message": message, "all_emotions": emotions})

@app.route("/history", methods=["GET"])
def get_history():
    auth_err = require_auth()
    if auth_err:
        return auth_err
    uid = session["user"]["id"]
    token = session["user"].get("access_token")
    try:
        params = {
            "select": "text,emotion,message,created_at",
            "user_id": f"eq.{uid}",
            "order": "created_at.desc",
            "limit": "50",
        }
        r = rest_get("emotions", params=params, token=token)
        if r.status_code == 200:
            return jsonify(r.json())
        logging.error("Failed to fetch history: %s %s", r.status_code, r.text)
        return jsonify({"error": "Failed to fetch history", "details": r.text}), 500
    except Exception as e:
        logging.exception("Failed to fetch history")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
