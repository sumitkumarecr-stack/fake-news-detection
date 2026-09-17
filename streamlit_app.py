from datetime import datetime
from pathlib import Path
import sqlite3

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DB_PATH = BASE_DIR / "predictions.db"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train_model.py` and commit "
            "model/fake_news_model.pkl and model/tfidf_vectorizer.pkl."
        )
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def init_db():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def clean_text(text):
    return " ".join(str(text).lower().split())


def predict_news(text, model, vectorizer):
    features = vectorizer.transform([clean_text(text)])
    prediction = str(model.predict(features)[0]).upper()
    confidence = float(max(model.predict_proba(features)[0]) * 100)
    return prediction, confidence


def save_prediction(text, prediction, confidence):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO predictions(text, prediction, confidence, created_at) "
            "VALUES (?, ?, ?, ?)",
            (text, prediction, confidence, datetime.now().isoformat(timespec="seconds")),
        )


def load_history():
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT prediction, confidence, text, created_at "
            "FROM predictions ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return [dict(row) for row in rows]


st.set_page_config(page_title="TruthLens AI", page_icon="📰", layout="centered")
st.title("TruthLens AI")
st.caption("A TF-IDF and Logistic Regression classifier for learned news-text patterns")
st.info("This model is not a fact-checking authority. Verify important claims with reliable sources.")

try:
    model, vectorizer = load_artifacts()
    init_db()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

news_text = st.text_area(
    "News text",
    placeholder="Paste at least 20 characters of a headline or article...",
    height=220,
)

if st.button("Analyze news", type="primary", use_container_width=True):
    text = news_text.strip()
    if len(text) < 20:
        st.warning("Please enter at least 20 characters of news text.")
    else:
        prediction, confidence = predict_news(text, model, vectorizer)
        save_prediction(text, prediction, confidence)
        st.session_state["last_result"] = prediction, confidence

if "last_result" in st.session_state:
    prediction, confidence = st.session_state["last_result"]
    st.subheader(f"Prediction: {prediction}")
    st.metric("Model confidence", f"{confidence:.2f}%")
    st.progress(min(confidence / 100, 1.0))

st.divider()
st.subheader("Recent predictions")
history = load_history()
if not history:
    st.caption("No predictions yet.")
else:
    for item in history:
        label = f"{item['prediction']} · {item['confidence']:.2f}% · {item['created_at']}"
        with st.expander(label):
            st.write(item["text"])