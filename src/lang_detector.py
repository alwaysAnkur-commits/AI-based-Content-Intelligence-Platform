# src/lang_detector.py
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# This is the modern FastText-replacement: a HF-hosted language ID model with 95%+ accuracy on 20+ languages
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

LANG_ROUTING = {
    "en": "no_translation_needed", "hi": "translate_then_summarize",
    "es": "translate_then_summarize", "fr": "translate_then_summarize",
    "de": "translate_then_summarize"
}

def detect_language(text: str) -> dict:
    if not text or len(text.strip()) < 10:
        return {"lang": "unknown", "confidence": 0.0}
    result = lang_detector(text[:512])[0]
    return {"lang": result["label"], "confidence": round(result["score"], 4)}

def detect_and_route_batch(records: list) -> pd.DataFrame:
    rows = []
    for rec in tqdm(records, desc="Detecting languages"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        detection = detect_language(text)
        route = LANG_ROUTING.get(detection["lang"], "translate_then_summarize")
        rows.append({"doc_id": rec.get("doc_id", ""), "detected_lang": detection["lang"],
                      "confidence": detection["confidence"], "routed_pipeline": route})
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/language_detection.csv", index=False)
    print(df["detected_lang"].value_counts())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    detect_and_route_batch(df.to_dict(orient="records"))