# src/summarizer.py
from transformers import pipeline
from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def summarize_text(text: str, max_len: int = 130, min_len: int = 30) -> str:
    if len(text.split()) < 50:
        return text  # too short to summarize meaningfully
    truncated = text[:4000]  # BART-large-cnn's effective input limit
    result = summarizer(truncated, max_length=max_len, min_length=min_len, do_sample=False)
    return result[0]["summary_text"]

def evaluate_summary(generated: str, reference: str) -> dict:
    """Compares generated summary against a reference (e.g. the article's own first paragraph as a proxy)."""
    scores = scorer.score(reference, generated)
    return {"rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4)}

def summarize_corpus_batch(records: list) -> pd.DataFrame:
    rows = []
    for rec in tqdm(records, desc="Summarizing"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        summary = summarize_text(text)
        # Use first 2 sentences of original as a rough reference for ROUGE (no labeled refs available)
        reference = ". ".join(text.split(". ")[:2])
        rouge = evaluate_summary(summary, reference) if reference else {}
        rows.append({"doc_id": rec.get("doc_id", ""), "summary": summary,
                      "original_length": len(text.split()), "summary_length": len(summary.split()),
                      **rouge})
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/summaries.csv", index=False)
    print(f"Summarized {len(df)} docs. Avg ROUGE-1: {df.get('rouge1', pd.Series([0])).mean():.3f}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    summarize_corpus_batch(df.to_dict(orient="records")[:50])