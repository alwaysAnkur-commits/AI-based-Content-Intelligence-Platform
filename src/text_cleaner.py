import re
import unicodedata
import ftfy
from bs4 import BeautifulSoup
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Regex patterns
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
MULTI_SPACE_RE = re.compile(r'[ \t]+')
MULTI_NEWLINE_RE = re.compile(r'\n{3,}')
URL_RE = re.compile(r'https?://\S+|www\.\S+')

def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text)

def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    return unicodedata.normalize(form, text)

def remove_control_chars(text: str) -> str:
    return CONTROL_CHAR_RE.sub("", text)

def collapse_whitespace(text: str) -> str:
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()

def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)

def clean_text(text: str, remove_url: bool = True, lowercase: bool = False) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = fix_encoding(text)
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    if remove_url:
        text = remove_urls(text)
    text = collapse_whitespace(text)
    if lowercase:
        text = text.lower()
    return text

def load_registry_map(registry_path="data/processed/document_registry.json"):
    if not Path(registry_path).exists():
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return {
        (v["title"].strip().lower(), v["source"].strip().lower()): v["doc_id"]
        for v in registry.values()
    }

if __name__ == "__main__":
    df = pd.read_csv("data/processed/deduped_articles.csv")
    records = df.to_dict(orient="records")

    # Load registry for doc_id mapping
    registry_map = load_registry_map()
    for r in records:
        title_key = str(r.get("title", "")).strip().lower()
        source_key = str(r.get("source", "")).strip().lower()
        r["doc_id"] = registry_map.get((title_key, source_key), r.get("doc_id", ""))

    # Apply cleaning
    df["doc_id"] = [r.get("doc_id", "") for r in records]
    df["body_clean"] = df["body"].apply(lambda x: clean_text(str(x)))
    df["title_clean"] = df["title"].apply(lambda x: clean_text(str(x), lowercase=False))

    # Append to single cleaned file, deduplicating by doc_id
    out_path = Path("data/processed/cleaned_articles.csv")
    if out_path.exists():
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["doc_id"], keep="last")
    df.to_csv(out_path, index=False)
    print(f"Updated {out_path} with {len(df)} total cleaned documents.")
