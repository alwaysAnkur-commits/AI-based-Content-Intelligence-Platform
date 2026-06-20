import pytest
import sys
sys.path.insert(0, "src")
from src.quality_checker import check_document_quality

GOOD_DOC = {
    "doc_id": "abc123",
    "title": "Tech company raises $50 million in Series B funding",
    "body": "A Mumbai-based AI startup announced today that it has raised $50 million in a Series B funding round led by Sequoia Capital, with participation from multiple angel investors.",
    "source": "TechCrunch",
    "url": "https://techcrunch.com/2025/01/01/startup-raises-funding"
}

def test_good_document_passes_and_scores_high():
    result = check_document_quality(GOOD_DOC)
    assert result["status"] == "PASS"
    assert result["quality_score"] >= 80

def test_missing_body_causes_failure():
    doc = {**GOOD_DOC, "body": ""}
    result = check_document_quality(doc)
    assert "MISSING_FIELD:body" in result["issues"] or result["quality_score"] < 60

def test_invalid_url_reduces_score():
    doc = {**GOOD_DOC, "url": "not-a-url"}
    result = check_document_quality(doc)
    base = check_document_quality(GOOD_DOC)
    assert result["quality_score"] < base["quality_score"]

def test_score_never_goes_below_zero():
    doc = {"doc_id": "", "title": "", "body": "", "source": "", "url": ""}
    result = check_document_quality(doc)
    assert result["quality_score"] >= 0