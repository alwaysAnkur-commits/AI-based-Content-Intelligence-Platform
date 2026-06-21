import pandas as pd
import json
import chardet
from pathlib import Path
from datetime import datetime

def load_registry_map(registry_path="data/processed/document_registry.json"):
    """Load registry and build a normalized (title, source) → doc_id map."""
    if not Path(registry_path).exists():
        print("⚠️ Registry file not found. doc_id will remain blank.")
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return {
        (v["title"].strip().lower(), v["source"].strip().lower()): v["doc_id"]
        for v in registry.values()
    }

def check_document_quality(record: dict, registry_map: dict) -> dict:
    """
    Score a single document 0–100 based on quality criteria.
    Deductions: missing fields, encoding issues, empty body, too short.
    """
    score = 100
    issues = []

    # Required fields check
    required_fields = ["title", "body", "source", "url"]
    for field in required_fields:
        val = record.get(field, "")
        if not val or str(val).strip() == "" or str(val) == "nan":
            score -= 20
            issues.append(f"MISSING_FIELD:{field}")

    # Body length check
    body = str(record.get("body", ""))
    if len(body) < 50:
        score -= 25
        issues.append("BODY_TOO_SHORT")
    elif len(body) < 200:
        score -= 10
        issues.append("BODY_SHORT")

    # Encoding check — detect non‑UTF8 content
    try:
        body.encode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        score -= 15
        issues.append("ENCODING_ERROR")

    # URL validation
    url = str(record.get("url", ""))
    if not url.startswith("http"):
        score -= 10
        issues.append("INVALID_URL")

    # Title quality
    title = str(record.get("title", ""))
    if len(title) < 5:
        score -= 10
        issues.append("TITLE_TOO_SHORT")

    # Lookup doc_id from registry
    title_key = title.strip().lower()
    source_key = str(record.get("source", "")).strip().lower()
    doc_id = registry_map.get((title_key, source_key), "")

    return {
        "doc_id": doc_id,
        "source": record.get("source", ""),
        "quality_score": max(0, score),
        "status": "PASS" if score >= 60 else "FAIL",
        "issues": "|".join(issues) if issues else "none",
        "checked_at": datetime.now().isoformat(),
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S")
    }

def run_quality_batch(records: list) -> pd.DataFrame:
    """Run quality checks on all documents and maintain cumulative CSV + versioned HTML report."""
    registry_map = load_registry_map()
    results = [check_document_quality(r, registry_map) for r in records]
    df = pd.DataFrame(results)

    # Enforce consistent column schema
    expected_cols = [
        "doc_id", "source", "quality_score", "status",
        "issues", "checked_at", "batch_id"
    ]
    df = df.reindex(columns=expected_cols)

    # Summary stats
    pass_rate = (df["status"] == "PASS").mean() * 100
    avg_score = df["quality_score"].mean()

    print(f"\nQuality Report:")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  Average score: {avg_score:.1f}/100")
    print(f"  Documents below 60: {(df['quality_score'] < 60).sum()}")

    # Save HTML report with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = df.style \
        .background_gradient(subset=["quality_score"], cmap="RdYlGn") \
        .set_caption(f"Data Quality Report — {datetime.now().strftime('%Y-%m-%d')}") \
        .to_html()

    Path("data/processed").mkdir(exist_ok=True)
    html_path = Path(f"data/processed/batch_quality_report_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Append to CSV log safely
    csv_path = Path("data/processed/quality_scores.csv")
    df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False
    )

    print(f"Saved: {html_path}")
    print(f"Appended results to: {csv_path}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/deduped_articles.csv")
    run_quality_batch(df.to_dict(orient="records"))
