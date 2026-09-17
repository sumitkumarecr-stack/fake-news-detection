"""
Train the Fake News Detection model.

Expected dataset:
data/Fake.csv
data/True.csv

Each CSV should contain a 'text' column. A 'title' column is optional.
The included sample files are only for testing the complete application.
For meaningful accuracy, replace them with a larger labeled dataset.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

def read_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    def combine(df):
        if "text" not in df.columns:
            raise ValueError("CSV must contain a 'text' column.")
        if "title" in df.columns:
            return (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
        return df["text"].fillna("").astype(str)

    fake_df = pd.DataFrame({"text": combine(fake), "label": "FAKE"})
    true_df = pd.DataFrame({"text": combine(true), "label": "REAL"})

    data = pd.concat([fake_df, true_df], ignore_index=True)
    data["text"] = data["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["text"].str.len() >= 20].drop_duplicates("text")

    return data

def main():
    data = read_dataset()

    if len(data) < 20 or data["label"].nunique() < 2:
        raise ValueError("Dataset is too small. Add more FAKE and REAL examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    joblib.dump(classifier, MODEL_DIR / "fake_news_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

    print("\nSaved:")
    print(MODEL_DIR / "fake_news_model.pkl")
    print(MODEL_DIR / "tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
