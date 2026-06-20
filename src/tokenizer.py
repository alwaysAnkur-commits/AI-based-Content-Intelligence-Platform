import spacy
from langdetect import detect, DetectorFactory
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm

# Fix langdetect random seed for consistent results
DetectorFactory.seed = 42

# Load models once at module level — NEVER inside a function
_models = {}

def get_model(lang: str) -> Optional[spacy.Language]:
    """Lazy-load spaCy models — load only when first needed."""
    global _models
    model_map = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "xx": "xx_ent_wiki_sm",  # multilingual fallback
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
    """Detect language code. Returns 'en', 'es', 'hi', 'fr', etc."""
    try:
        return detect(text[:500])  # use first 500 chars for speed
    except Exception:
        return "en"  # default to English on failure

HINDI_STOPWORDS = {"और", "में", "की", "है", "के", "को", "से", "का", "यह", "वह", "हैं", "भी", "पर"}

def tokenize_and_lemmatize(text: str, lang: str = None) -> Dict:
    """
    Tokenize, lemmatize, and remove stopwords.
    Auto-detects language if lang not provided.
    Returns structured dict with tokens, lemmas, and POS tags.
    """
    if not text or len(text.strip()) < 5:
        return {"lang": "unknown", "tokens": [], "lemmas": [], "pos_tags": []}
    
    lang = lang or detect_language(text)
    
    # Hindi — spaCy doesn't have an official hi model; do basic whitespace tokenization
    if lang == "hi":
        tokens = [t for t in text.split() if t not in HINDI_STOPWORDS and len(t) > 1]
        return {"lang": "hi", "tokens": tokens, "lemmas": tokens, "pos_tags": []}
    
    # Route to spaCy model
    route_lang = lang if lang in ["en", "es"] else "xx"
    nlp = get_model(route_lang)
    
    doc = nlp(text[:5000])  # cap at 5000 chars to prevent memory issues
    
    tokens = []
    lemmas = []
    pos_tags = []
    
    for token in doc:
        # Skip: stopwords, punctuation, spaces, very short tokens
        if token.is_stop or token.is_punct or token.is_space or len(token.text) < 2:
            continue
        tokens.append(token.text)
        lemmas.append(token.lemma_.lower())
        pos_tags.append(token.pos_)
    
    return {
        "lang": lang,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "token_count": len(tokens)
    }

def process_corpus(records: list, text_col: str = "body_clean") -> pd.DataFrame:
    """
    Process all documents using nlp.pipe() for speed.
    nlp.pipe() is 3-5x faster than calling nlp() in a loop.
    """
    # Separate by language first for batch efficiency
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
    df.to_csv("data/processed/tokenized_corpus.csv", index=False)
    
    # Language distribution summary
    print("\nLanguage Distribution:")
    print(df["detected_lang"].value_counts())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_articles.csv").fillna("")
    process_corpus(df.to_dict(orient="records"))