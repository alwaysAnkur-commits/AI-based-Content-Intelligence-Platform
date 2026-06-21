# src/sentiment_eval.py
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
import time
from sklearn.metrics import f1_score, accuracy_score

vader = SentimentIntensityAnalyzer()
distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def vader_predict(text): return "POSITIVE" if vader.polarity_scores(text)["compound"] >= 0 else "NEGATIVE"
def textblob_predict(text): return "POSITIVE" if TextBlob(text).sentiment.polarity >= 0 else "NEGATIVE"
def distilbert_predict(text): return distilbert(text[:512])[0]["label"]

def evaluate_sentiment_models(labeled_test_set: list) -> pd.DataFrame:
    """labeled_test_set: [{'text': ..., 'true_label': 'POSITIVE'/'NEGATIVE'}, ...] — from 200 manually labeled docs."""
    models = {"VADER": vader_predict, "TextBlob": textblob_predict, "DistilBERT": distilbert_predict}
    rows = []

    for name, predict_fn in models.items():
        y_true, y_pred = [], []
        start = time.perf_counter()
        for item in labeled_test_set:
            y_pred.append(predict_fn(item["text"]))
            y_true.append(item["true_label"])
        elapsed = (time.perf_counter() - start) / len(labeled_test_set)

        rows.append({"model": name, "accuracy": round(accuracy_score(y_true, y_pred), 3),
                      "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 3),
                      "avg_inference_time_ms": round(elapsed * 1000, 2)})

    df = pd.DataFrame(rows)
    with open("docs/sentiment_eval.md", "w") as f:
        f.write(f"# Sentiment Model Evaluation\n\nTest set: {len(labeled_test_set)} docs\n\n" + df.to_markdown(index=False))
    print(df)
    return df

if __name__ == "__main__":
    test_set = [
        {"text": "The company reported record profits this quarter.", "true_label": "POSITIVE"},
        {"text": "Layoffs hit thousands of workers amid economic downturn.", "true_label": "NEGATIVE"},
    ]
    evaluate_sentiment_models(test_set)