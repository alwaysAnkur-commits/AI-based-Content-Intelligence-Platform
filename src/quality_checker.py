import pandas as pd
import chardet
from pathlib import Path
from datetime import datetime

def check_document_quality(record: dict) -> dict:
    """
    Score a single document 0-100 based on quality criteria.
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
    
    # Encoding check — detect non-UTF8 content
    try:
        body.encode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        score -= 15
        issues.append("ENCODING_ERROR")
    
    # Duplicate URL / empty URL
    url = str(record.get("url", ""))
    if not url.startswith("http"):
        score -= 10
        issues.append("INVALID_URL")
    
    # Title quality
    title = str(record.get("title", ""))
    if len(title) < 5:
        score -= 10
        issues.append("TITLE_TOO_SHORT")
    
    return {
        "doc_id": record.get("doc_id", ""),
        "source": record.get("source", ""),
        "quality_score": max(0, score),
        "status": "PASS" if score >= 60 else "FAIL",
        "issues": "|".join(issues) if issues else "none",
        "checked_at": datetime.now().isoformat()
    }

def run_quality_batch(records: list) -> pd.DataFrame:
    """Run quality checks on all documents and generate an HTML report."""
    results = [check_document_quality(r) for r in records]
    df = pd.DataFrame(results)
    
    # Summary stats
    pass_rate = (df["status"] == "PASS").mean() * 100
    avg_score = df["quality_score"].mean()
    
    print(f"\nQuality Report:")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  Average score: {avg_score:.1f}/100")
    print(f"  Documents below 60: {(df['quality_score'] < 60).sum()}")
    
    # Save HTML report using pandas styling
    html = df.style \
        .background_gradient(subset=["quality_score"], cmap="RdYlGn") \
        .set_caption(f"Data Quality Report — {datetime.now().strftime('%Y-%m-%d')}") \
        .to_html()
    
    Path("data/processed").mkdir(exist_ok=True)
    with open("data/processed/batch_quality_report.html", "w") as f:
        f.write(html)
    df.to_csv("data/processed/quality_scores.csv", index=False)
    print("Saved: data/processed/batch_quality_report.html")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/deduped_articles.csv")
    run_quality_batch(df.to_dict(orient="records"))