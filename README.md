# 📰 Fake News Detection System

A GitHub-ready machine-learning web application that classifies news text as **FAKE** or **REAL** using TF-IDF feature extraction and Logistic Regression.

> **Important:** A classifier predicts patterns learned from its training data. It is not a substitute for professional fact-checking or verification of a news story.

## ✨ Features

- FAKE / REAL news classification
- Confidence score
- TF-IDF with unigram + bigram features
- Logistic Regression classifier
- Flask REST API
- Streamlit web application
- Responsive web interface
- SQLite prediction history
- Model evaluation during training
- Health-check endpoint
- Easy dataset replacement
- Clean GitHub repository structure

## 🧠 Architecture

```text
News Text
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
FAKE / REAL + Confidence
   ↓
Flask Web App
   ↓
SQLite History
```

## 📁 Project Structure

```text
fake-news-detection/
├── app.py
├── streamlit_app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── Fake.csv
│   └── True.csv
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── screenshots/
    └── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sumitkumarecr-stack/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This creates:

```text
model/fake_news_model.pkl
model/tfidf_vectorizer.pkl
```

### 5. Start the web application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000

### Run with Streamlit

The Streamlit entry point is `streamlit_app.py`:

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including both files in `model/`:

   ```bash
   git add streamlit_app.py requirements.txt .gitignore model/*.pkl
   git commit -m "Add Streamlit deployment"
   git push origin main
   ```

   If the model files were previously ignored, use `git add -f model/*.pkl`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. Select **New app**, choose this repository and branch, and set the main file to
   `streamlit_app.py`.
4. Select **Deploy**. Streamlit Cloud installs `requirements.txt` and starts the app.

The SQLite history is suitable for a demonstration, but Streamlit Cloud storage is
ephemeral and can reset when the app restarts. Use a hosted database if history must
survive redeployments.
```

## 📊 Dataset

The repository contains a **small synthetic demonstration dataset** so the application can be run immediately.

For a serious academic/project evaluation, replace `data/Fake.csv` and `data/True.csv` with a sufficiently large, documented dataset. The CSV files should contain:

```text
text
```

and may optionally contain:

```text
title,text
```

Recommended sources include established public research datasets. Document the exact dataset name, license, preprocessing, and train/test split in your final report.

## 🔌 API

### POST `/predict`

Request:

```json
{
  "text": "News article text goes here..."
}
```

Response:

```json
{
  "prediction": "REAL",
  "confidence": 87.42
}
```

### GET `/history`

Returns the latest 20 predictions.

### GET `/health`

Returns application/model status.

## 🧪 Command-line Prediction

After training:

```bash
python predict.py
```

## 🎓 IBM Project / Viva Points

### Problem
False or misleading information can spread rapidly through online platforms. Manual verification is time-consuming, so an NLP-based classifier can assist with initial screening.

### Objective
Build a machine-learning system that analyzes the linguistic patterns of a news article and predicts whether it resembles examples labeled FAKE or REAL in the training dataset.

### Why TF-IDF?
TF-IDF converts text into numerical features based on the importance of words and phrases within the dataset.

### Why Logistic Regression?
It is a strong, interpretable baseline for high-dimensional text classification and is computationally efficient.

### Limitations
- The model learns from the dataset rather than independently verifying facts.
- Dataset bias can affect predictions.
- Satire, opinion, breaking news, and unfamiliar events can be difficult.
- Confidence is model confidence, not proof that a story is true.

## 🔮 Future Scope

- Transformer models such as BERT
- Multilingual fake-news detection
- Source credibility analysis
- Claim extraction and evidence retrieval
- Explainable AI with highlighted influential terms
- News URL ingestion
- Human fact-checker workflow
- Model monitoring and drift detection

## 📜 License

MIT License. See `LICENSE`.

## 👨‍💻 Author

Replace this section with your name, college, GitHub profile, and project details before submission.
