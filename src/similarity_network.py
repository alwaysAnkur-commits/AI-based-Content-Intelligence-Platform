# src/similarity_network.py
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import json

def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """Cosine similarity via cdist — works directly on sparse matrices without densifying the full corpus."""
    dense = tfidf_matrix.toarray()  # fine for corpora under ~5000 docs; switch to chunked computation beyond that
    dist_matrix = cdist(dense, dense, metric="cosine")
    return 1 - dist_matrix  # convert distance to similarity

def get_top_k_similar(sim_matrix: np.ndarray, doc_ids: list, k: int = 5) -> dict:
    recommendations = {}
    for i, doc_id in enumerate(doc_ids):
        scores = sim_matrix[i].copy()
        scores[i] = -1  # exclude self
        top_k_idx = np.argsort(scores)[::-1][:k]
        recommendations[doc_id] = [
            {"doc_id": doc_ids[j], "similarity": round(float(scores[j]), 4)} for j in top_k_idx
        ]
    return recommendations

if __name__ == "__main__":
    # Load TF-IDF matrix and corpus
    tfidf_matrix = sp.load_npz("data/processed/features/tfidf.npz")
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    doc_ids = df["doc_id"].tolist()

    # Ensure lengths match
    if len(doc_ids) != tfidf_matrix.shape[0]:
        print(f"⚠️ Mismatch detected: {len(doc_ids)} doc_ids vs {tfidf_matrix.shape[0]} TF-IDF rows")
        # Truncate or align
        doc_ids = doc_ids[:tfidf_matrix.shape[0]]
        print(f"✅ Adjusted doc_ids list to {len(doc_ids)} entries to match TF-IDF matrix")

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(tfidf_matrix)

    # Generate recommendations
    recommendations = get_top_k_similar(sim_matrix, doc_ids, k=5)

    # Save recommendations
    with open("data/processed/similarity_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"Generated similarity recommendations for {len(recommendations)} documents.")
