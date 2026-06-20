import re
import unicodedata
import ftfy
from bs4 import BeautifulSoup
from typing import Optional

# Control character regex — removes invisible chars like \x00-\x1F except \n and \t
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
MULTI_SPACE_RE = re.compile(r'[ \t]+')
MULTI_NEWLINE_RE = re.compile(r'\n{3,}')
URL_RE = re.compile(r'https?://\S+|www\.\S+')

def fix_encoding(text: str) -> str:
    """Use ftfy to fix broken unicode / mojibake automatically."""
    return ftfy.fix_text(text)

def strip_html(text: str) -> str:
    """Remove all HTML tags and return clean readable text."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """
    NFKC: best for NLP — converts fullwidth chars, ligatures, etc.
    to their standard equivalents. E.g. ﬁ → fi, ① → 1
    """
    return unicodedata.normalize(form, text)

def remove_control_chars(text: str) -> str:
    return CONTROL_CHAR_RE.sub("", text)

def collapse_whitespace(text: str) -> str:
    """Normalize all whitespace — tabs to spaces, multiple spaces to one."""
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()

def remove_urls(text: str) -> bool:
    """Optionally remove URLs — set to False if URL text is meaningful."""
    return URL_RE.sub(" ", text)

def clean_text(text: str, remove_url: bool = True, lowercase: bool = False) -> str:
    """
    Master cleaning function — applies all steps in the CORRECT ORDER.
    Order matters: fix encoding → strip HTML → normalize unicode →
    remove control chars → remove URLs (optional) → collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Fix broken encoding first — ftfy handles latin-1/cp1252 mojibake
    text = fix_encoding(text)
    # Step 2: Strip HTML tags (must come before unicode normalization)
    text = strip_html(text)
    # Step 3: Normalize unicode to NFKC
    text = normalize_unicode(text)
    # Step 4: Remove invisible control characters
    text = remove_control_chars(text)
    # Step 5: Remove URLs if requested
    if remove_url:
        text = remove_urls(text)
    # Step 6: Collapse whitespace — always last
    text = collapse_whitespace(text)
    # Step 7: Lowercase (optional — keep original case for NER)
    if lowercase:
        text = text.lower()
    
    return text

def benchmark_cleaner(sample_docs: list, n: int = 200) -> dict:
    """
    Benchmark the cleaner on n documents.
    Returns noise removal rate and timing stats.
    """
    import time
    sample = sample_docs[:n]
    start = time.perf_counter()
    cleaned = [clean_text(d.get("body", "")) for d in sample]
    elapsed = time.perf_counter() - start
    
    # Noise removed = avg reduction in length (HTML tags, whitespace, etc.)
    orig_lens = [len(d.get("body", "")) for d in sample]
    clean_lens = [len(c) for c in cleaned]
    noise_removed = [(o - c) / o * 100 if o > 0 else 0 
                     for o, c in zip(orig_lens, clean_lens)]
    avg_noise = sum(noise_removed) / len(noise_removed) if noise_removed else 0
    
    print(f"Benchmarked {n} docs in {elapsed:.2f}s ({n/elapsed:.0f} docs/sec)")
    print(f"Average noise removed: {avg_noise:.1f}%")
    return {"docs_per_sec": n / elapsed, "avg_noise_removed_pct": avg_noise}

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed/deduped_articles.csv")
    records = df.to_dict(orient="records")
    benchmark_cleaner(records, n=min(200, len(records)))
    
    # Apply cleaning to all docs
    df["body_clean"] = df["body"].apply(lambda x: clean_text(str(x)))
    df["title_clean"] = df["title"].apply(lambda x: clean_text(str(x), lowercase=False))
    df.to_csv("data/processed/cleaned_articles.csv", index=False)
    print(f"Saved {len(df)} cleaned documents.")