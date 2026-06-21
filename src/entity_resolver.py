# src/entity_resolver.py
from rapidfuzz import fuzz, process
import pandas as pd

KNOWLEDGE_BASE = {
    "google": "Google LLC", "google inc.": "Google LLC", "google inc": "Google LLC",
    "alphabet": "Alphabet Inc.", "alphabet inc.": "Alphabet Inc.",
    "meta": "Meta Platforms, Inc.", "facebook": "Meta Platforms, Inc.", "fb": "Meta Platforms, Inc.",
}

def resolve_entity(entity_text: str, canonical_list: list, threshold: int = 85) -> dict:
    normalized = entity_text.lower().strip()
    if normalized in KNOWLEDGE_BASE:
        return {"original": entity_text, "resolved": KNOWLEDGE_BASE[normalized],
                 "confidence": 100, "method": "knowledge_base"}

    if canonical_list:
        match, score, _ = process.extractOne(entity_text, canonical_list, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            return {"original": entity_text, "resolved": match, "confidence": score, "method": "fuzzy_match"}

    return {"original": entity_text, "resolved": entity_text, "confidence": 100, "method": "new_canonical"}

def build_resolution_table(entities_df: pd.DataFrame) -> pd.DataFrame:
    unique_entities = entities_df["entity_text"].unique().tolist()
    canonical_list = list(set(KNOWLEDGE_BASE.values()))

    resolutions = []
    for entity in unique_entities:
        result = resolve_entity(entity, canonical_list)
        resolutions.append(result)
        if result["method"] == "new_canonical" and result["resolved"] not in canonical_list:
            canonical_list.append(result["resolved"])

    df = pd.DataFrame(resolutions)
    df.to_csv("data/processed/entity_resolution_mapping.csv", index=False)
    print(f"Resolved {len(df)} mentions into {df['resolved'].nunique()} canonical entities.")
    return df

if __name__ == "__main__":
    entities_df = pd.read_csv("data/processed/entities.csv")
    build_resolution_table(entities_df)