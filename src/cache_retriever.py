# src/cached_retriever.py
from collections import OrderedDict
from hybrid_search import hybrid_search

class CachedRetriever:
    def __init__(self, collection, bm25, doc_ids, max_size: int = 100):
        self.collection = collection
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def search(self, query: str, top_k: int = 10) -> list:
        cache_key = f"{query.lower().strip()}::{top_k}"
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)  # mark as recently used
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1
        results = hybrid_search(query, self.collection, self.bm25, self.doc_ids, top_k=top_k)
        self.cache[cache_key] = results
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # evict least-recently-used
        return results

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_ratio": round(self.hits / total, 3) if total else 0}

if __name__ == "__main__":
    from vector_db_setup import get_chroma_client
    from hybrid_search import build_bm25_index
    import pandas as pd

    client = get_chroma_client()
    collection = client.get_collection("content_corpus")
    tokenized_df = pd.read_csv("data/processed/tokenized_corpus.csv").fillna("")
    bm25, doc_ids = build_bm25_index(tokenized_df)

    retriever = CachedRetriever(collection, bm25, doc_ids)
    retriever.search("AI startups")
    retriever.search("AI startups")  # cache hit
    retriever.search("climate policy")
    print(retriever.get_stats())