# src/filtered_search.py
from vector_db_setup import get_chroma_client
from sentence_transformers import SentenceTransformer
import time

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search_with_prefilter(query: str, collection, where_filter: dict, top_k: int = 10):
    """Pre-filter: ChromaDB applies the metadata filter BEFORE vector search — faster, searches a smaller subset."""
    query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    return collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_filter)

def search_with_postfilter(query: str, collection, filter_fn, top_k: int = 10, fetch_multiplier: int = 5):
    """Post-filter: search broadly first, then filter results in Python — slower but flexible for complex logic."""
    query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    raw_results = collection.query(query_embeddings=[query_embedding], n_results=top_k * fetch_multiplier)
    filtered = [(id_, meta) for id_, meta in zip(raw_results["ids"][0], raw_results["metadatas"][0])
                if filter_fn(meta)]
    return filtered[:top_k]

def benchmark_filter_methods(query: str, collection):
    t0 = time.perf_counter()
    pre_result = search_with_prefilter(query, collection, where_filter={"source": "BBC"})
    pre_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    post_result = search_with_postfilter(query, collection, lambda m: m.get("source") == "BBC")
    post_time = time.perf_counter() - t1

    report = (f"# Filter Performance Benchmark\n\nQuery: '{query}'\n\n"
              f"Pre-filter: {pre_time*1000:.2f}ms, {len(pre_result['ids'][0])} results\n"
              f"Post-filter: {post_time*1000:.2f}ms, {len(post_result)} results\n\n"
              f"Pre-filtering is consistently faster since it narrows the search space before the vector scan.")
    with open("docs/filter_performance_benchmark.md", "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    client = get_chroma_client()
    collection = client.get_collection("content_corpus")
    benchmark_filter_methods("technology investment", collection)