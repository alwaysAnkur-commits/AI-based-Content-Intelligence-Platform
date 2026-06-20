import pytest
import sys
sys.path.insert(0, "src")
from src.confidence_router import route_document

def test_high_confidence_auto_accepted():
    result = route_document("doc_001", "news", 0.92)
    assert result["routing_status"] == "auto_accepted"

def test_mid_confidence_flagged():
    result = route_document("doc_002", "blog", 0.72)
    assert result["routing_status"] == "flagged_for_review"

def test_low_confidence_rejected():
    result = route_document("doc_003", "invoice", 0.45)
    assert result["routing_status"] == "rejected"

def test_boundary_exactly_85_percent():
    """Exactly at the boundary should be auto_accepted."""
    result = route_document("doc_004", "email", 0.85)
    assert result["routing_status"] == "auto_accepted"

def test_result_always_has_reason_code():
    for conf in [0.95, 0.70, 0.30]:
        result = route_document("doc_x", "news", conf)
        assert "reason" in result and len(result["reason"]) > 0