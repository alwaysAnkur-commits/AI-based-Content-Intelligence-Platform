import uuid, json
from datetime import datetime
from pathlib import Path

REGISTRY_PATH = "data/processed/document_registry.json"

def load_registry():
    if Path(REGISTRY_PATH).exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def register_document(title, source, url="", processing_status="ingested", extra=None):
    registry = load_registry()
    doc_id = str(uuid.uuid4())
    entry = {
        "doc_id": doc_id,
        "title": title,
        "source": source,
        "url": url,
        "source_tag": source.lower().replace(" ", "_"),
        "ingested_at": datetime.now().isoformat(),
        "processing_status": processing_status,
        "extra": extra or {}
    }
    registry[doc_id] = entry
    save_registry(registry)
    return doc_id

def update_status(doc_id, new_status):
    registry = load_registry()
    if doc_id in registry:
        registry[doc_id]["processing_status"] = new_status
        registry[doc_id]["updated_at"] = datetime.now().isoformat()
        save_registry(registry)

def bulk_register(records):
    """Register all scraped/API records into the tracker."""
    ids = []
    for rec in records:
        doc_id = register_document(
            title=rec.get("title", ""),
            source=rec.get("source", "unknown"),
            url=rec.get("url", ""),
            extra={"scrape_timestamp": rec.get("scrape_timestamp", "")}
        )
        ids.append(doc_id)
    print(f"Registered {len(ids)} documents into registry.")
    return ids

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed/deduped_articles.csv")
    bulk_register(df.to_dict(orient="records"))