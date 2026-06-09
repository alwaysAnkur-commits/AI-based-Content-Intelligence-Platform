from transformers import pipeline
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm

# The 6 categories from the task spec
CATEGORIES = ["news", "blog", "research paper", "invoice", "legal", "email"]

# Load once — this model is ~1.6GB, do NOT reload inside a loop
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # CPU; change to 0 if GPU available
)

def classify_document(text: str, multi_label: bool = False) -> dict:
    """
    Classify a single document into one of 6 categories.
    Returns label + confidence scores for all categories.
    multi_label=True allows multiple labels to fire simultaneously.
    """
    if not text or len(text.strip()) < 20:
        return {"predicted_label": "unknown", "confidence": 0.0, "all_scores": {}}
    
    # Truncate to 1024 chars for speed — BART has a 1024 token limit anyway
    truncated = text[:1024]
    
    result = classifier(truncated, candidate_labels=CATEGORIES, multi_label=multi_label)
    
    # result structure: {"labels": [...], "scores": [...], "sequence": "..."}
    scores_dict = dict(zip(result["labels"], result["scores"]))
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    
    return {
        "predicted_label": top_label,
        "confidence": round(top_score, 4),
        "all_scores": {k: round(v, 4) for k, v in scores_dict.items()}
    }

def classify_batch(records: list, text_col: str = "body") -> pd.DataFrame:
    """Classify a list of document dicts and return a DataFrame with classification results."""
    results = []
    for rec in tqdm(records, desc="Classifying documents"):
        text = rec.get(text_col, "") or rec.get("title", "")
        classification = classify_document(text)
        results.append({
            "doc_id": rec.get("doc_id", ""),
            "title": rec.get("title", ""),
            "source": rec.get("source", ""),
            "predicted_label": classification["predicted_label"],
            "confidence": classification["confidence"],
            **{f"score_{k.replace(' ', '_')}": v 
               for k, v in classification["all_scores"].items()}
        })
    
    df = pd.DataFrame(results)
    df.to_csv("data/processed/classification_report.csv", index=False)
    print(f"\nClassification complete. {len(df)} documents classified.")
    print(df["predicted_label"].value_counts())
    return df

# Logistic Regression FALLBACK if model is too slow or GPU unavailable
def classify_with_tfidf_fallback(records: list) -> pd.DataFrame:
    """
    Fast fallback using keyword rules — not ML, but enough for testing.
    Use only when BART is too slow on your machine.
    """
    KEYWORD_MAP = {
        "invoice": ["invoice", "amount due", "payment", "bill", "total"],
        "legal": ["whereas", "hereinafter", "plaintiff", "defendant", "court"],
        "email": ["dear", "regards", "sincerely", "subject:", "from:", "to:"],
        "research paper": ["abstract", "methodology", "conclusion", "references", "doi"],
        "news": ["reported", "according to", "sources say", "breaking"],
        "blog": ["i think", "in my opinion", "today i", "tutorial", "how to"]
    }
    results = []
    for rec in records:
        text = (rec.get("title", "") + " " + rec.get("body", "")).lower()
        scores = {cat: sum(1 for kw in kws if kw in text) 
                  for cat, kws in KEYWORD_MAP.items()}
        predicted = max(scores, key=scores.get)
        results.append({
            "doc_id": rec.get("doc_id", ""),
            "predicted_label": predicted,
            "confidence": 0.5,  # fixed fallback confidence
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/deduped_articles.csv")
    records = df.to_dict(orient="records")
    classify_batch(records[:50])  # start with 50 to test speed