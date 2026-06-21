import textstat
import pandas as pd
from collections import Counter
import re
from datetime import datetime

def compute_stats(text: str) -> dict:
    """Compute comprehensive statistics for a single document."""
    if not text or len(text.strip()) < 10:
        return _empty_stats()
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences)
    unique_words = set(words)
    
    # Vocabulary richness: Type-Token Ratio
    ttr = len(unique_words) / word_count if word_count > 0 else 0
    
    # Average word length
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    
    # Readability scores via textstat
    flesch_ease = textstat.flesch_reading_ease(text)
    flesch_grade = textstat.flesch_kincaid_grade(text)
    fog_index = textstat.gunning_fog(text)
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "unique_word_count": len(unique_words),
        "avg_word_length": round(avg_word_len, 2),
        "vocabulary_richness_ttr": round(ttr, 4),
        "flesch_reading_ease": round(flesch_ease, 2),
        "flesch_kincaid_grade": round(flesch_grade, 2),
        "gunning_fog_index": round(fog_index, 2),
        "avg_sentence_length": round(word_count / sentence_count, 1) if sentence_count > 0 else 0
    }

def _empty_stats() -> dict:
    return {k: 0 for k in [
        "word_count", "char_count", "sentence_count", "unique_word_count",
        "avg_word_length", "vocabulary_richness_ttr", "flesch_reading_ease",
        "flesch_kincaid_grade", "gunning_fog_index", "avg_sentence_length"
    ]}

def generate_corpus_stats(records: list) -> pd.DataFrame:
    stats_rows = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for rec in records:
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        stats = compute_stats(text)
        stats["doc_id"] = rec.get("doc_id", "")
        stats["source"] = rec.get("source", "")
        stats["run_id"] = run_id
        stats_rows.append(stats)
    
    df = pd.DataFrame(stats_rows)
    out_path = "data/processed/doc_stats.csv"
    
    # Append mode with deduplication
    try:
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["doc_id"], keep="last")
    except FileNotFoundError:
        pass
    
    df.to_csv(out_path, index=False)
    print(f"Updated {out_path} with {len(df)} total documents.")
    
    # Aggregate summary
    numeric_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64"]]
    summary = df[numeric_cols].describe()
    print("\nCorpus Statistics Summary:")
    print(summary.round(2))
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_articles.csv").fillna("")
    generate_corpus_stats(df.to_dict(orient="records"))
