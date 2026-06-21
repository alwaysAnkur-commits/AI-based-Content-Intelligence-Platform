# src/bertopic_modeler.py
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd

def build_bertopic(docs: list, min_topic_size: int = 10) -> BERTopic:
    """Build a BERTopic model using sentence embeddings."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

def reduce_and_save(topic_model: BERTopic, docs: list, target_topics: int = 12) -> BERTopic:
    """Merge over-granular topics down to a target count — BERTopic often over-splits initially."""
    topic_model.reduce_topics(docs=docs, nr_topics=target_topics)
    return topic_model

def compare_with_lda(bertopic_info: pd.DataFrame, lda_csv_path: str = "data/processed/lda_topics.csv"):
    """Compare BERTopic results with LDA and save a markdown report."""
    lda_df = pd.read_csv(lda_csv_path)
    report = [
        "# BERTopic vs LDA Comparison\n",
        f"BERTopic discovered {bertopic_info['Topic'].nunique()} topics (excluding outlier topic -1).",
        f"LDA discovered {lda_df['dominant_topic'].nunique()} topics.\n",
        "## Key Differences",
        "- BERTopic uses dense embeddings + clustering (HDBSCAN), capturing semantic similarity directly.",
        "- LDA uses bag-of-words probabilistic modeling — purely word co-occurrence statistics.",
        "- BERTopic can leave documents unassigned (-1 'outlier' topic); LDA always assigns every document.",
        "- BERTopic topics tend to be more semantically coherent for short/noisy text.",
    ]
    with open("data/processed/bertopic_vs_lda_comparison.md", "w") as f:
        f.write("\n".join(report))
    print("Saved comparison report.")

if __name__ == "__main__":
    # Load corpus
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    docs = df["body_clean"].tolist()

    # Build BERTopic model
    model, topics, probs = build_bertopic(docs)

    # Reduce topics to target count
    model = reduce_and_save(model, docs, target_topics=12)

    # Get topic info
    topic_info = model.get_topic_info()

    # Save per-document topic assignments
    result_df = pd.DataFrame({"doc_id": df["doc_id"], "bertopic_topic": model.topics_})
    result_df.to_csv("data/processed/bertopic_topics.csv", index=False)

    # Compare with LDA
    compare_with_lda(topic_info)

    # Print preview of topics
    print(topic_info.head(10))
