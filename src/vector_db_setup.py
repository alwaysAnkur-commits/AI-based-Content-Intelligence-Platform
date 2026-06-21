# src/vector_db_setup.py
import chromadb
import numpy as np
import pandas as pd

def get_chroma_client():
    return chromadb.PersistentClient(path="data/chroma_db")  # persists across restarts — never use in-memory Client() for this

def build_metadata(corpus_df: pd.DataFrame, sentiment_df: pd.DataFrame = None,
                    entities_df: pd.DataFrame = None) -> list:
    metadata_list = []
    for _, row in corpus_df.iterrows():
        meta = {"title": str(row.get("title", ""))[:200], "source": str(row.get("source", "")),
                "date": str(row.get("date", "")), "category": str(row.get("predicted_label", "unknown"))}
        if sentiment_df is not None:
            sent_row = sentiment_df[sentiment_df["doc_id"] == row["doc_id"]]
            meta["sentiment"] = str(sent_row["sentiment_label"].iloc[0]) if not sent_row.empty else "unknown"
        if entities_df is not None:
            entity_count = len(entities_df[entities_df["doc_id"] == row["doc_id"]])
            meta["entity_count"] = entity_count
        metadata_list.append(meta)
    return metadata_list

def index_corpus(embeddings: np.ndarray, doc_ids: list, metadata: list, collection_name: str = "content_corpus"):
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists — adding to it.")
    except Exception:
        collection = client.create_collection(collection_name)

    batch_size = 100
    for i in range(0, len(doc_ids), batch_size):
        collection.add(
            embeddings=embeddings[i:i+batch_size].tolist(),
            ids=[str(d) for d in doc_ids[i:i+batch_size]],
            metadatas=metadata[i:i+batch_size]
        )
    print(f"Indexed {collection.count()} documents into ChromaDB.")
    return collection

if __name__ == "__main__":
    embeddings = np.load("data/processed/embeddings/corpus_embeddings.npy")
    doc_ids = np.load("data/processed/embeddings/doc_ids.npy", allow_pickle=True).tolist()
    corpus_df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")

    sentiment_df = pd.read_csv("data/processed/sentiment_scores.csv") if __import__("os").path.exists("data/processed/sentiment_scores.csv") else None
    entities_df = pd.read_csv("data/processed/entities.csv") if __import__("os").path.exists("data/processed/entities.csv") else None

    metadata = build_metadata(corpus_df, sentiment_df, entities_df)
    index_corpus(embeddings, doc_ids, metadata)