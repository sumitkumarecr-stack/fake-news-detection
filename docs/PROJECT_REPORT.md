# Fake News Detection — Project Report

## 1. Introduction

Fake news detection is an NLP classification problem in which a machine-learning model learns patterns associated with labeled examples of misleading and legitimate news.

## 2. Objectives

- Build a text-classification pipeline.
- Convert news text into numerical TF-IDF features.
- Train a Logistic Regression classifier.
- Provide predictions through a web interface.
- Store recent predictions for demonstration and analysis.

## 3. Technologies

| Component | Technology |
|---|---|
| Language | Python |
| NLP | TF-IDF |
| ML | Logistic Regression |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Model serialization | Joblib |

## 4. Methodology

1. Collect labeled FAKE and REAL news.
2. Clean and combine title/text fields.
3. Remove duplicate/very short records.
4. Split data into training and test sets using stratification.
5. Fit TF-IDF vectorization on training data.
6. Train Logistic Regression.
7. Evaluate on the held-out test set.
8. Save model artifacts.
9. Serve predictions through Flask.

## 5. Evaluation

Run:

```bash
python train_model.py
```

Record the displayed accuracy, precision, recall, F1-score, and confusion matrix in the final report after using your actual dataset.

Do not claim a model accuracy that was not measured on your actual dataset.

## 6. Conclusion

The application demonstrates an end-to-end NLP machine-learning workflow from data preprocessing to deployment. It is designed as an educational project and should be extended with a larger, well-documented dataset and stronger language models for higher-quality research.
