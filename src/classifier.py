from transformers import pipeline
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# The 6 categories from the task spec
CATEGORIES = ["news", "blog", "research paper", "invoice", "legal", "email"]

# Load once — this model is ~1.6GB, do NOT reload inside a loop
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # CPU; change to 0 if GPU available
)

def load_registry_map(registry_path="data/processed/document_registry.json"):
    """Load registry and build a normalized (title, source) → doc_id map."""
    if not Path(registry_path).exists():
        print("⚠️ Registry file not found. doc_id will remain blank.")
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    # Normalize title and source for better matching
    return {
        (v["title"].strip().lower(), v["source"].strip().lower()): v["doc_id"]
        for v in registry.values()
    }

def classify_document(text: str, multi_label: bool = False) -> dict:
    """Classify a single document into one of 6 categories."""
    if not text or len(text.strip()) < 20:
        return {"predicted_label": "unknown", "confidence": 0.0, "all_scores": {}}
    truncated = text[:1024]
    result = classifier(truncated, candidate_labels=CATEGORIES, multi_label=multi_label)
    scores_dict = dict(zip(result["labels"], result["scores"]))
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return {
        "predicted_label": top_label,
        "confidence": round(top_score, 4),
        "all_scores": {k: round(v, 4) for k, v in scores_dict.items()}
    }

def classify_batch(records: list, text_col: str = "body") -> pd.DataFrame:
    """Classify a list of document dicts and append results to classification_report.csv."""
    registry_map = load_registry_map()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for rec in tqdm(records, desc="Classifying documents"):
        text = rec.get(text_col, "") or rec.get("title", "")
        classification = classify_document(text)
        # Normalize lookup keys
        title_key = rec.get("title", "").strip().lower()
        source_key = rec.get("source", "").strip().lower()
        doc_id = registry_map.get((title_key, source_key), "")
        if not doc_id:
            print(f"⚠️ Missing doc_id for: {rec.get('title')} ({rec.get('source')})")

        results.append({
            "batch_id": batch_id,
            "doc_id": doc_id,
            "title": rec.get("title", ""),
            "source": rec.get("source", ""),
            "predicted_label": classification["predicted_label"],
            "confidence": classification["confidence"],
            **{f"score_{k.replace(' ', '_')}": v 
               for k, v in classification["all_scores"].items()}
        })

    df = pd.DataFrame(results)
    out_path = Path("data/processed/classification_report.csv")
    df.to_csv(
        out_path,
        mode="a",  # append mode
        header=not out_path.exists(),
        index=False
    )

    print(f"\nClassification complete. {len(df)} documents classified.")
    print(df["predicted_label"].value_counts())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/deduped_articles.csv")
    records = df.to_dict(orient="records")
    classify_batch(records[:50])  # start with 50 to test speed
