from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import sqlite3
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DB_PATH = BASE_DIR / "predictions.db"

app = Flask(__name__)

MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def clean_text(text):
    return " ".join(str(text).lower().split())

def predict_news(text):
    if model is None or vectorizer is None:
        raise RuntimeError("Model not found. Run: python train_model.py")
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities) * 100)
    return str(prediction).upper(), confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()

    if len(text) < 20:
        return jsonify({"error": "Please enter at least 20 characters of news text."}), 400

    try:
        prediction, confidence = predict_news(text)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO predictions(text, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
                (text, prediction, confidence, datetime.now().isoformat(timespec="seconds"))
            )
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 2)
        })
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

@app.get("/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, text, prediction, confidence, created_at "
            "FROM predictions ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return jsonify([dict(row) for row in rows])

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None and vectorizer is not None
    })

load_artifacts()
init_db()

if __name__ == "__main__":
    app.run(debug=True)
