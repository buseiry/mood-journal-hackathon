from flask import Flask, request, jsonify, session
from flask_cors import CORS
import psycopg2
import psycopg2.extras
from flask_bcrypt import Bcrypt
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")
CORS(app, supports_credentials=True)
bcrypt = Bcrypt(app)

# PostgreSQL connection
conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ---------------- Auth Routes ----------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    try:
        cur.execute("INSERT INTO users (email, password) VALUES (%s, %s) RETURNING id, email", (email, hashed))
        user = cur.fetchone()
        conn.commit()
        session["user_id"] = user["id"]
        return jsonify({"user": user})
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"error": "Email already exists"}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    cur.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    if user and bcrypt.check_password_hash(user["password"], password):
        session["user_id"] = user["id"]
        return jsonify({"user": {"id": user["id"], "email": user["email"]}})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out"})

@app.route("/user")
def get_user():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    cur.execute("SELECT id, email FROM users WHERE id=%s", (uid,))
    user = cur.fetchone()
    return jsonify({"user": user})

# ---------------- Mood Routes ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Authentication required"}), 401
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Dummy emotion analysis
    import random
    emotions = ["joy", "sadness", "anger", "fear", "disgust", "neutral", "surprise"]
    emotion_label = random.choice(emotions)
    emotion_score = random.randint(50, 100)

    cur.execute("""
        INSERT INTO emotions (user_id, text, emotion_label, emotion_score)
        VALUES (%s, %s, %s, %s)
        RETURNING id, text, emotion_label, emotion_score, created_at
    """, (uid, text, emotion_label, emotion_score))
    entry = cur.fetchone()
    conn.commit()

    return jsonify({
        "emotion": entry["emotion_label"],
        "score": entry["emotion_score"],
        "text": entry["text"],
        "created_at": entry["created_at"],
        "all_emotions": [{"label": entry["emotion_label"], "score": entry["emotion_score"]/100}]
    })

@app.route("/history")
def history():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Authentication required"}), 401
    cur.execute("SELECT * FROM emotions WHERE user_id=%s ORDER BY created_at DESC", (uid,))
    entries = cur.fetchall()
    return jsonify(entries)

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
