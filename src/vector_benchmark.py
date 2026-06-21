# # src/vector_benchmark.py
# import faiss
# import numpy as np
# import time
# import tracemalloc

# def benchmark_flat(embeddings: np.ndarray, queries: np.ndarray, k: int = 10) -> dict:
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     tracemalloc.start()
#     t0 = time.perf_counter()
#     distances, indices = index.search(queries, k)
#     latency = (time.perf_counter() - t0) / len(queries) * 1000
#     _, peak_mem = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     return {"method": "FlatL2 (brute-force)", "p95_latency_ms": round(latency, 3),
#             "memory_mb": round(peak_mem / 1e6, 2), "indices": indices}

# def benchmark_ivf(embeddings: np.ndarray, queries: np.ndarray, nlist: int = 100, k: int = 10) -> dict:
#     dim = embeddings.shape[1]
#     quantizer = faiss.IndexFlatL2(dim)
#     index = faiss.IndexIVFFlat(quantizer, dim, nlist)
#     index.train(embeddings)
#     index.add(embeddings)
#     index.nprobe = 10  # number of clusters searched per query — speed/accuracy tradeoff knob

#     tracemalloc.start()
#     t0 = time.perf_counter()
#     distances, indices = index.search(queries, k)
#     latency = (time.perf_counter() - t0) / len(queries) * 1000
#     _, peak_mem = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     return {"method": f"IVF{nlist}", "p95_latency_ms": round(latency, 3),
#             "memory_mb": round(peak_mem / 1e6, 2), "indices": indices}

# def compute_recall_at_10(flat_indices: np.ndarray, ivf_indices: np.ndarray) -> float:
#     """Flat search is ground truth (exact); recall@10 = how many of IVF's top-10 match Flat's top-10."""
#     matches = sum(len(set(f) & set(i)) for f, i in zip(flat_indices, ivf_indices))
#     return matches / (len(flat_indices) * 10)

# if __name__ == "__main__":
#     embeddings = np.load("data/processed/embeddings/corpus_embeddings.npy").astype("float32")
#     queries = embeddings[:50]  # use first 50 docs as query proxies

#     flat_result = benchmark_flat(embeddings, queries)
#     ivf100 = benchmark_ivf(embeddings, queries, nlist=100)
#     ivf500 = benchmark_ivf(embeddings, queries, nlist=min(500, len(embeddings) // 10 or 1))

#     recall_100 = compute_recall_at_10(flat_result["indices"], ivf100["indices"])
#     recall_500 = compute_recall_at_10(flat_result["indices"], ivf500["indices"])

#     report = f"""# Vector Search Benchmark

# Corpus size: {len(embeddings)} vectors, {embeddings.shape[1]} dimensions

# | Method | p95 Latency (ms) | Memory (MB) | Recall@10 |
# |--------|-------------------|-------------|-----------|
# | {flat_result['method']} | {flat_result['p95_latency_ms']} | {flat_result['memory_mb']} | 1.0000 (ground truth) |
# | {ivf100['method']} | {ivf100['p95_latency_ms']} | {ivf100['memory_mb']} | {recall_100:.4f} |
# | {ivf500['method']} | {ivf500['p95_latency_ms']} | {ivf500['memory_mb']} | {recall_500:.4f} |
# """
#     with open("docs/vector_benchmark.md", "w") as f:
#         f.write(report)
#     print(report)

# src/vector_benchmark.py
import faiss
import numpy as np
import time

def benchmark_ivf(embeddings: np.ndarray, queries: np.ndarray, nlist: int = 100):
    """
    Build and benchmark an IVF index.
    Automatically adjusts nlist so it's never larger than the number of training points.
    """
    d = embeddings.shape[1]
    n_vectors = embeddings.shape[0]

    # ✅ Fix: ensure nlist <= number of vectors
    if nlist > n_vectors:
        print(f"⚠️ Requested nlist={nlist} but only {n_vectors} vectors available. Adjusting nlist to {n_vectors}.")
        nlist = max(1, n_vectors)  # at least 1 cluster

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    # Train and add embeddings
    index.train(embeddings)
    index.add(embeddings)

    # Benchmark search
    start = time.perf_counter()
    D, I = index.search(queries, k=5)
    elapsed = time.perf_counter() - start
    print(f"IVF-{nlist} search time: {elapsed:.4f}s for {len(queries)} queries")
    return index

if __name__ == "__main__":
    # Example: load embeddings and queries
    embeddings = np.random.rand(35, 128).astype("float32")  # 35 vectors, 128-dim
    queries = np.random.rand(5, 128).astype("float32")      # 5 queries

    # This will auto-adjust nlist if too large
    ivf100 = benchmark_ivf(embeddings, queries, nlist=100)
