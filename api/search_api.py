# api/search_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys, pandas as pd
sys.path.insert(0, "src")
from vector_db_setup import get_chroma_client
from hybrid_search import build_bm25_index, hybrid_search

app = FastAPI(title="Content Intelligence Search API", version="1.0")

client = get_chroma_client()
collection = client.get_collection("content_corpus")
tokenized_df = pd.read_csv("data/processed/tokenized_corpus.csv").fillna("")
bm25, doc_ids = build_bm25_index(tokenized_df)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    source_filter: Optional[str] = None

class SearchResult(BaseModel):
    doc_id: str
    fused_score: float
    dense_score: float
    sparse_score: float

class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResult]

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = hybrid_search(request.query, collection, bm25, doc_ids, top_k=request.top_k)
    return SearchResponse(query=request.query, result_count=len(results), results=results)

@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": collection.count()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)