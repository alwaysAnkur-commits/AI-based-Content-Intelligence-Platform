# src/sentiment_analyzer.py
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

sentiment_pipe = pipeline("sentiment-analysis",
                            model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

def analyze_sentiment_batch(records: list, batch_size: int = 32) -> pd.DataFrame:
    results = []
    texts = [str(r.get("body_clean", "") or r.get("body", ""))[:512] for r in records]

    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment analysis"):
        batch = texts[i:i + batch_size]
        batch_results = sentiment_pipe(batch)
        for rec, res in zip(records[i:i + batch_size], batch_results):
            results.append({"doc_id": rec.get("doc_id", ""), "source": rec.get("source", ""),
                             "sentiment_label": res["label"], "sentiment_confidence": round(res["score"], 4)})

    df = pd.DataFrame(results)
    df.to_csv("data/processed/sentiment_scores.csv", index=False)
    print(f"\n{df['sentiment_label'].value_counts()}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    analyze_sentiment_batch(df.to_dict(orient="records"))