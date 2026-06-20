import numpy as np
import scipy.sparse as sp
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("data/processed/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_tfidf(corpus: list, max_features: int = 10000, ngram_range=(1, 2)) -> dict:
    """
    Fit and transform corpus with TF-IDF.
    ngram_range=(1,2) captures single words AND two-word phrases (bigrams).
    Returns vectorizer, matrix, and feature names.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,           # ignore terms appearing in fewer than 2 docs
        max_df=0.95,        # ignore terms in more than 95% of docs (too common)
        sublinear_tf=True,  # log normalization — reduces impact of very high TF
        strip_accents="unicode",
        analyzer="word"
    )
    
    matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.3f}")
    
    # Save matrix and vectorizer
    sp.save_npz(OUTPUT_DIR / "tfidf.npz", matrix)
    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")
    
    # Save feature names
    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump({"tfidf_features": feature_names[:500]}, f, indent=2)  # save top 500
    
    print(f"Saved: tfidf.npz, tfidf_vectorizer.joblib")
    return {"vectorizer": vectorizer, "matrix": matrix, "features": feature_names}

def build_count_vectorizer(corpus: list, max_features: int = 5000) -> dict:
    """Bag-of-Words count matrix — complement to TF-IDF."""
    vec = CountVectorizer(max_features=max_features, min_df=2, max_df=0.95)
    matrix = vec.fit_transform(corpus)
    sp.save_npz(OUTPUT_DIR / "count_vectors.npz", matrix)
    joblib.dump(vec, OUTPUT_DIR / "count_vectorizer.joblib")
    print(f"Count matrix shape: {matrix.shape}")
    return {"vectorizer": vec, "matrix": matrix}

def get_top_terms_per_doc(tfidf_matrix, feature_names: list, n: int = 10) -> list:
    """Extract top n TF-IDF terms for each document — useful for debugging."""
    top_terms = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray()[0]
        top_idx = row.argsort()[::-1][:n]
        terms = [(feature_names[j], round(row[j], 4)) for j in top_idx if row[j] > 0]
        top_terms.append(terms)
    return top_terms

if __name__ == "__main__":
    df = pd.read_csv("data/processed/tokenized_corpus.csv").fillna("")
    corpus = df["lemmas"].tolist()  # use lemmatized text for better features
    
    tfidf_result = build_tfidf(corpus)
    count_result = build_count_vectorizer(corpus)
    
    # Show top terms for first 3 documents
    top = get_top_terms_per_doc(tfidf_result["matrix"], tfidf_result["features"])
    for i in range(min(3, len(top))):
        print(f"\nDoc {i} top terms: {top[i][:5]}")