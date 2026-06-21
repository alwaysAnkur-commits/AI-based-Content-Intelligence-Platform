# src/ner_pipeline.py
import spacy
from spacy import displacy
import pandas as pd
from tqdm import tqdm
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

# Target 8 entity types from spaCy's built-in label set
TARGET_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT", "LAW"}

def add_domain_patterns(nlp_model):
    """Add rule-based patterns BEFORE statistical NER for deterministic domain matches."""
    ruler = nlp_model.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "PRODUCT", "pattern": [{"LOWER": "series"}, {"TEXT": {"IN": ["A", "B", "C", "D"]}}]},
        {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": r"^\$?\d+(\.\d+)?[MmKkBb]?$"}}]},
    ]
    ruler.add_patterns(patterns)
    return nlp_model

nlp = add_domain_patterns(nlp)

def extract_entities(text: str) -> list:
    doc = nlp(text[:100000])
    return [{"text": ent.text, "label": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char}
            for ent in doc.ents if ent.label_ in TARGET_ENTITY_TYPES]

def visualize_entities(text: str, output_path: str = "data/processed/ner_visualization.html"):
    doc = nlp(text)
    html = displacy.render(doc, style="ent", page=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved entity visualization to {output_path}")

def process_corpus_ner(records: list) -> pd.DataFrame:
    all_entities = []
    for rec in tqdm(records, desc="Extracting entities"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        for ent in extract_entities(text):
            all_entities.append({
                "doc_id": rec.get("doc_id", ""),
                "entity_text": ent["text"],
                "entity_label": ent["label"],
                "start_char": ent["start_char"],
                "end_char": ent["end_char"]
            })

    df = pd.DataFrame(all_entities)

    # ✅ Append mode: write header only if file doesn't exist
    output_path = Path("data/processed/entities.csv")
    df.to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False
    )

    print(f"\nExtracted {len(df)} entities across {len(records)} documents.")
    print(df["entity_label"].value_counts())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    process_corpus_ner(df.to_dict(orient="records"))
    visualize_entities(df.iloc[0]["body_clean"])
