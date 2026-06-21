# src/hybrid_search.py
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from vector_db_setup import get_chroma_client
import pandas as pd, json

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_bm25_index(tokenized_corpus_df: pd.DataFrame) -> tuple:
    tokenized_docs = [str(t).split() for t in tokenized_corpus_df["lemmas"].fillna("")]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_corpus_df["doc_id"].tolist()

def dense_search(query: str, collection, top_k: int = 10) -> dict:
    query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return {doc_id: 1 - dist for doc_id, dist in zip(results["ids"][0], results["distances"][0])}

def sparse_search(query: str, bm25: BM25Okapi, doc_ids: list, top_k: int = 10) -> dict:
    scores = bm25.get_scores(query.lower().split())
    top_idx = scores.argsort()[::-1][:top_k]
    max_score = max(scores[top_idx]) if len(top_idx) and scores[top_idx[0]] > 0 else 1
    return {doc_ids[i]: scores[i] / max_score for i in top_idx}  # normalize to 0-1 for fair fusion

def hybrid_search(query: str, collection, bm25: BM25Okapi, doc_ids: list,
                   alpha: float = 0.5, top_k: int = 10) -> list:
    """alpha=0.5 means equal weight to dense (semantic) and sparse (keyword) scores."""
    dense_scores = dense_search(query, collection, top_k=top_k * 2)
    sparse_scores = sparse_search(query, bm25, doc_ids, top_k=top_k * 2)

    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    fused = []
    for doc_id in all_ids:
        d_score = dense_scores.get(doc_id, 0)
        s_score = sparse_scores.get(doc_id, 0)
        fused_score = alpha * d_score + (1 - alpha) * s_score
        fused.append({"doc_id": doc_id, "fused_score": round(fused_score, 4),
                      "dense_score": round(d_score, 4), "sparse_score": round(s_score, 4)})
    return sorted(fused, key=lambda x: -x["fused_score"])[:top_k]

if __name__ == "__main__":
    client = get_chroma_client()
    collection = client.get_collection("content_corpus")
    tokenized_df = pd.read_csv("data/processed/tokenized_corpus.csv").fillna("")
    bm25, doc_ids = build_bm25_index(tokenized_df)

    results = hybrid_search("artificial intelligence investment trends", collection, bm25, doc_ids)
    with open("data/processed/hybrid_search_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    for r in results[:5]:
        print(r)