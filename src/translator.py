# src/translator.py
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd, json
from tqdm import tqdm

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
SUPPORTED_LANGS = ["en", "hi", "es", "fr", "de"]

def chunk_text(text: str, max_tokens: int = 480) -> list:
    """M2M100 has a 512-token limit; chunk on word boundaries with margin for special tokens."""
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang == tgt_lang or not text.strip():
        return text
    tokenizer.src_lang = src_lang
    translated_chunks = []
    for chunk in chunk_text(text):
        encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
        translated_chunks.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
    return " ".join(translated_chunks)

def translate_corpus_batch(records: list, target_lang: str = "en", source_lang: str = "hi") -> list:
    results = []
    for rec in tqdm(records, desc=f"Translating {source_lang}->{target_lang}"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))[:3000]
        translated = translate_text(text, source_lang, target_lang)
        results.append({"doc_id": rec.get("doc_id", ""), "source_lang": source_lang,
                         "target_lang": target_lang, "original_text": text, "translated_text": translated})
    with open("data/processed/translated_corpus.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Translated {len(results)} documents.")
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    # Test with a small sample first — full corpus on CPU is slow
    translate_corpus_batch(df.to_dict(orient="records")[:10], target_lang="en", source_lang="hi")