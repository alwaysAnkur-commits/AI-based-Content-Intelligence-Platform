import spacy
from langdetect import detect, DetectorFactory
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm
from pathlib import Path

DetectorFactory.seed = 42
_models = {}

def get_model(lang: str) -> Optional[spacy.Language]:
    global _models
    model_map = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "xx": "xx_ent_wiki_sm",
    }
    if lang not in _models:
        model_name = model_map.get(lang, "xx_ent_wiki_sm")
        try:
            _models[lang] = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Falling back to xx_ent_wiki_sm")
            if "xx" not in _models:
                _models["xx"] = spacy.load("xx_ent_wiki_sm")
            _models[lang] = _models["xx"]
    return _models[lang]

def detect_language(text: str) -> str:
    try:
        return detect(text[:500])
    except Exception:
        return "en"

HINDI_STOPWORDS = {"और","में","की","है","के","को","से","का","यह","वह","हैं","भी","पर"}

def tokenize_and_lemmatize(text: str, lang: str = None) -> Dict:
    if not text or len(text.strip()) < 5:
        return {"lang": "unknown", "tokens": [], "lemmas": [], "pos_tags": []}
    lang = lang or detect_language(text)
    if lang == "hi":
        tokens = [t for t in text.split() if t not in HINDI_STOPWORDS and len(t) > 1]
        return {"lang": "hi", "tokens": tokens, "lemmas": tokens, "pos_tags": []}
    route_lang = lang if lang in ["en","es"] else "xx"
    nlp = get_model(route_lang)
    doc = nlp(text[:5000])
    tokens, lemmas, pos_tags = [], [], []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space or len(token.text) < 2:
            continue
        tokens.append(token.text)
        lemmas.append(token.lemma_.lower())
        pos_tags.append(token.pos_)
    return {"lang": lang,"tokens": tokens,"lemmas": lemmas,"pos_tags": pos_tags,"token_count": len(tokens)}

def process_corpus(records: list, text_col: str = "body_clean") -> pd.DataFrame:
    results = []
    for rec in tqdm(records, desc="Tokenizing"):
        text = str(rec.get(text_col, "") or rec.get("body", ""))
        lang = detect_language(text)
        result = tokenize_and_lemmatize(text, lang=lang)
        results.append({
            "doc_id": rec.get("doc_id", ""),
            "detected_lang": result["lang"],
            "token_count": len(result["tokens"]),
            "tokens": " ".join(result["tokens"]),
            "lemmas": " ".join(result["lemmas"]),
        })
    df = pd.DataFrame(results)

    # Append to single tokenized file, deduplicating by doc_id
    out_path = Path("data/processed/tokenized_corpus.csv")
    if out_path.exists():
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["doc_id"], keep="last")
    df.to_csv(out_path, index=False)
    print(f"Updated {out_path} with {len(df)} total tokenized documents.")

    print("\nLanguage Distribution:")
    print(df["detected_lang"].value_counts())
    print(f"\nAverage token count per document: {df['token_count'].mean():.1f}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_articles.csv").fillna("")
    process_corpus(df.to_dict(orient="records"))
