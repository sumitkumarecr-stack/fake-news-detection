from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model" / "fake_news_model.pkl")
vectorizer = joblib.load(BASE_DIR / "model" / "tfidf_vectorizer.pkl")

text = input("Enter news text: ").strip()
X = vectorizer.transform([text])
prediction = model.predict(X)[0]
confidence = max(model.predict_proba(X)[0]) * 100

print(f"\nPrediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")
