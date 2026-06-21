# src/entity_summarizer.py
import pandas as pd
import json
from collections import Counter

def summarize_document_entities(doc_id, entities_df, triplets_df, resolution_df) -> dict:
    doc_entities = entities_df[entities_df["doc_id"] == doc_id]
    doc_triplets = triplets_df[triplets_df["doc_id"] == doc_id]
    resolution_map = dict(zip(resolution_df["original"], resolution_df["resolved"]))
    resolved_counts = Counter(resolution_map.get(e, e) for e in doc_entities["entity_text"])

    relationships = [{"subject": r["subject"], "relation": r["mapped_relation_type"], "object": r["object"]}
                      for _, r in doc_triplets.iterrows()]

    return {
        "doc_id": doc_id,
        "total_entity_mentions": len(doc_entities),
        "unique_entities": doc_entities["entity_text"].nunique(),
        "entities_by_type": doc_entities.groupby("entity_label")["entity_text"].apply(list).to_dict(),
        "top_entities": dict(resolved_counts.most_common(5)),
        "relationships": relationships,
        "relationship_count": len(relationships)
    }

def generate_all_summaries(entities_df, triplets_df, resolution_df, output_path="data/processed/entity_summaries.json"):
    doc_ids = entities_df["doc_id"].unique()
    summaries = [summarize_document_entities(d, entities_df, triplets_df, resolution_df) for d in doc_ids]
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)
    avg = sum(s["total_entity_mentions"] for s in summaries) / len(summaries)
    print(f"Generated {len(summaries)} summaries. Avg entities/doc: {avg:.1f}")
    return summaries

if __name__ == "__main__":
    entities_df = pd.read_csv("data/processed/entities.csv")
    triplets_df = pd.read_csv("data/processed/relation_triplets.csv")
    resolution_df = pd.read_csv("data/processed/entity_resolution_mapping.csv")
    generate_all_summaries(entities_df, triplets_df, resolution_df)
    