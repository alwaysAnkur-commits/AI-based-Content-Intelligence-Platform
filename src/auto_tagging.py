# src/auto_tagger.py
from keybert import KeyBERT
import pandas as pd

kw_model = KeyBERT(model="all-MiniLM-L6-v2")

def extract_keywords(text: str, top_n: int = 5) -> list:
    if len(text.split()) < 10:
        return []
    keywords = kw_model.extract_keywords(text[:2000], keyphrase_ngram_range=(1, 2),
                                          stop_words="english", top_n=top_n, use_mmr=True, diversity=0.5)
    return [kw for kw, score in keywords]

def generate_tags(record: dict, topic_keywords: str) -> list:
    text = str(record.get("body_clean", "") or record.get("body", ""))
    keybert_tags = extract_keywords(text, top_n=3)
    topic_tags = [t.strip() for t in str(topic_keywords).split(",")[:2]]
    combined = list(dict.fromkeys(keybert_tags + topic_tags))  # dedupe, preserve order
    return combined[:5]

def tag_corpus(corpus_df: pd.DataFrame, topics_df: pd.DataFrame) -> pd.DataFrame:
    merged = corpus_df.merge(topics_df[["doc_id", "topic_keywords"]], on="doc_id", how="left").fillna("")
    rows = []
    for _, row in merged.iterrows():
        tags = generate_tags(row.to_dict(), row.get("topic_keywords", ""))
        rows.append({"doc_id": row["doc_id"], "tags": "|".join(tags), "tag_count": len(tags)})
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/auto_tags.csv", index=False)
    print(f"Tagged {len(df)} documents.")
    return df

def evaluate_against_labels(tags_df: pd.DataFrame, labels_path: str = "data/test/tag_labels.csv"):
    """Optional — only run if you've built a manually labeled test set."""
    try:
        labels = pd.read_csv(labels_path)
        merged = tags_df.merge(labels, on="doc_id")
        merged["overlap"] = merged.apply(
            lambda r: len(set(r["tags"].split("|")) & set(str(r["true_tags"]).split("|"))) > 0, axis=1)
        accuracy = merged["overlap"].mean()
        print(f"Tag overlap accuracy: {accuracy*100:.1f}%")
    except FileNotFoundError:
        print(f"No label file at {labels_path} — skipping evaluation. Tagging still completed successfully.")

if __name__ == "__main__":
    corpus_df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    topics_df = pd.read_csv("data/processed/lda_topics.csv")
    tags_df = tag_corpus(corpus_df, topics_df)
    evaluate_against_labels(tags_df)