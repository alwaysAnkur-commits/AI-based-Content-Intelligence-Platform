import pytest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, "src")

# --- Tests for classifier.py ---

def test_classify_document_returns_expected_keys():
    """Classifier output must always have these 3 keys."""
    from src.classifier import classify_document
    # Mock the global 'classifier' pipeline to avoid loading 1.6GB model in tests
    with patch("classifier.classifier") as mock_clf:
        mock_clf.return_value = {
            "labels": ["news", "blog", "email"],
            "scores": [0.91, 0.06, 0.03],
            "sequence": "test text"
        }
        result = classify_document("Breaking news: AI startup raises $100M funding round.")
    
    assert "predicted_label" in result
    assert "confidence" in result
    assert "all_scores" in result

def test_classify_document_empty_text_returns_unknown():
    """Empty text should return 'unknown' without calling the model."""
    from src.classifier import classify_document
    result = classify_document("")
    assert result["predicted_label"] == "unknown"
    assert result["confidence"] == 0.0

def test_classify_document_confidence_range():
    """Confidence score must always be between 0 and 1."""
    from src.classifier import classify_document
    with patch("classifier.classifier") as mock_clf:
        mock_clf.return_value = {
            "labels": ["legal", "news"],
            "scores": [0.73, 0.27],
            "sequence": "hereinafter referred to as the plaintiff"
        }
        result = classify_document("Hereinafter referred to as the plaintiff.")
    assert 0.0 <= result["confidence"] <= 1.0