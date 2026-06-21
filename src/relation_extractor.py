# src/relation_extractor.py
import spacy
import pandas as pd
from tqdm import tqdm
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

RELATION_RULES = {
    "works_at": ["work", "employ", "join"],
    "acquired": ["acquire", "buy", "purchase"],
    "happened_on": ["occur", "happen"],
    "founded": ["found", "establish", "launch"],
    "located_in": ["base", "headquarter", "locate"],
}

def map_relation_type(relation_lemma: str) -> str:
    for rel_type, verbs in RELATION_RULES.items():
        if relation_lemma in verbs:
            return rel_type
    return "other"

def extract_relations(text: str) -> list:
    """SVO triplet extraction via dependency parsing: subject -nsubj-> VERB <-dobj/pobj- object."""
    doc = nlp(text[:100000])
    triplets = []

    for token in doc:
        if token.pos_ == "VERB":
            subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.rights if w.dep_ in ("dobj", "attr", "pobj")]
            for right in token.rights:
                if right.dep_ == "prep":
                    objects += [c for c in right.children if c.dep_ == "pobj"]

            for subj in subjects:
                for obj in objects:
                    subj_span = next((c for c in doc.noun_chunks if subj in c), subj)
                    obj_span = next((c for c in doc.noun_chunks if obj in c), obj)
                    triplets.append({
                        "subject": subj_span.text,
                        "relation": token.lemma_,
                        "object": obj_span.text,
                        "sentence": token.sent.text.strip()
                    })
    return triplets

def extract_relations_batch(records: list) -> pd.DataFrame:
    all_triplets = []
    for rec in tqdm(records, desc="Extracting relations"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        for t in extract_relations(text):
            t["doc_id"] = rec.get("doc_id", "")
            t["mapped_relation_type"] = map_relation_type(t["relation"])
            all_triplets.append(t)

    df = pd.DataFrame(all_triplets)

    # ✅ Deduplicate by doc_id + subject + relation + object
    df = df.drop_duplicates(subset=["doc_id", "subject", "relation", "object"], keep="last")

    # ✅ Append mode with deduplication against existing file
    output_path = Path("data/processed/relation_triplets.csv")
    try:
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["doc_id", "subject", "relation", "object"], keep="last")
        combined.to_csv(output_path, index=False)
        df = combined
    except FileNotFoundError:
        df.to_csv(output_path, index=False)

    print(f"\nExtracted {len(df)} unique triplets.")
    print(df["mapped_relation_type"].value_counts())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    extract_relations_batch(df.to_dict(orient="records"))
