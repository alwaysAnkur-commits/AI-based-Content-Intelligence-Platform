# src/aspect_sentiment.py
import spacy
from transformers import pipeline
import pandas as pd

nlp = spacy.load("en_core_web_sm")
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def extract_aspect_sentiment(text: str, target_entities: list) -> list:
    """Sentiment scoped to the sentence an entity appears in, not the whole document."""
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        for entity in target_entities:
            if entity.lower() in sent_text.lower() and len(sent_text) > 5:
                sentiment = sentiment_pipe(sent_text[:512])[0]
                results.append({"entity": entity, "sentence": sent_text,
                                 "sentiment_label": sentiment["label"], "sentiment_score": round(sentiment["score"], 4)})
    return results

def aggregate_aspect_sentiment(records: list, top_entities: list):
    all_results = []
    for rec in records:
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        for a in extract_aspect_sentiment(text, top_entities):
            a["doc_id"] = rec.get("doc_id", "")
            all_results.append(a)

    df = pd.DataFrame(all_results)
    summary = df.groupby("entity")["sentiment_label"].value_counts(normalize=True).mul(100).round(1).reset_index(name="percentage")
    df.to_csv("data/processed/aspect_sentiment_detail.csv", index=False)
    summary.to_csv("data/processed/aspect_sentiment_summary.csv", index=False)
    print(summary)
    return df, summary

if __name__ == "__main__":
    entities_df = pd.read_csv("data/processed/entities.csv")
    top_entities = entities_df["entity_text"].value_counts().head(20).index.tolist()
    corpus_df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    aggregate_aspect_sentiment(corpus_df.to_dict(orient="records"), top_entities)