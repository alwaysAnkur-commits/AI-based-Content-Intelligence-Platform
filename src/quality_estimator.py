# src/quality_estimator.py
from evaluate import load
import pandas as pd, json

bertscore = load("bertscore")

def estimate_translation_quality(original: str, translated: str) -> dict:
    """Reference-free proxy: round-trip semantic similarity via BERTScore against the original (cross-lingual model)."""
    if not translated.strip():
        return {"bertscore_f1": 0.0}
    result = bertscore.compute(predictions=[translated], references=[original],
                                model_type="distilbert-base-multilingual-cased")
    return {"bertscore_f1": round(result["f1"][0], 4)}

def estimate_summary_quality(original: str, summary: str) -> dict:
    """Compression ratio + BERTScore against original (no human reference needed)."""
    compression_ratio = len(summary.split()) / max(len(original.split()), 1)
    result = bertscore.compute(predictions=[summary], references=[original[:2000]], lang="en")
    return {"compression_ratio": round(compression_ratio, 3), "bertscore_f1": round(result["f1"][0], 4)}

def run_quality_estimation():
    rows = []
    try:
        with open("data/processed/translated_corpus.json") as f:
            translations = json.load(f)
        for t in translations:
            q = estimate_translation_quality(t["original_text"], t["translated_text"])
            rows.append({"doc_id": t["doc_id"], "type": "translation", **q})
    except FileNotFoundError:
        print("No translated_corpus.json found — skipping translation QE")

    try:
        summaries_df = pd.read_csv("data/processed/summaries.csv")
        corpus_df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
        merged = summaries_df.merge(corpus_df[["doc_id", "body_clean"]], on="doc_id", how="left")
        for _, row in merged.iterrows():
            q = estimate_summary_quality(str(row["body_clean"]), str(row["summary"]))
            rows.append({"doc_id": row["doc_id"], "type": "summary", **q})
    except FileNotFoundError:
        print("No summaries.csv found — skipping summary QE")

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/quality_estimates.csv", index=False)
    print(f"Quality estimation complete for {len(df)} items.")
    return df

if __name__ == "__main__":
    run_quality_estimation()