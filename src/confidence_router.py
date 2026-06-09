import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Thresholds per the task spec
AUTO_ACCEPT_THRESHOLD = 0.85
FLAG_LOWER_BOUND = 0.60

REASON_CODES = {
    "low_confidence": "Model confidence below 60% — likely ambiguous or noisy document",
    "borderline": "Confidence between 60-85% — human review recommended",
    "accepted": "High-confidence classification — auto-accepted"
}

def route_document(doc_id: str, predicted_label: str, confidence: float) -> dict:
    """
    Route a single document based on its confidence score.
    Returns routing decision with reason code.
    """
    if confidence >= AUTO_ACCEPT_THRESHOLD:
        status = "auto_accepted"
        reason = REASON_CODES["accepted"]
    elif confidence >= FLAG_LOWER_BOUND:
        status = "flagged_for_review"
        reason = REASON_CODES["borderline"]
    else:
        status = "rejected"
        reason = REASON_CODES["low_confidence"]
    
    return {
        "doc_id": doc_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "routing_status": status,
        "reason": reason,
        "routed_at": datetime.now().isoformat()
    }

def build_review_queue(classification_df: pd.DataFrame) -> dict:
    """
    Process all classified documents and build:
    - auto_accepted list
    - review_queue (flagged 60-85%)
    - rejected list
    """
    queue = {"auto_accepted": [], "review_queue": [], "rejected": []}
    summary = {"auto_accepted": 0, "flagged": 0, "rejected": 0}
    
    for _, row in classification_df.iterrows():
        decision = route_document(
            doc_id=row.get("doc_id", ""),
            predicted_label=row.get("predicted_label", ""),
            confidence=float(row.get("confidence", 0.0))
        )
        if decision["routing_status"] == "auto_accepted":
            queue["auto_accepted"].append(decision)
            summary["auto_accepted"] += 1
        elif decision["routing_status"] == "flagged_for_review":
            queue["review_queue"].append(decision)
            summary["flagged"] += 1
        else:
            queue["rejected"].append(decision)
            summary["rejected"] += 1
    
    queue["summary"] = summary
    queue["generated_at"] = datetime.now().isoformat()
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/review_queue.json", "w") as f:
        json.dump(queue, f, indent=2)
    
    print(f"\nRouting Summary:")
    print(f" Auto-accepted: {summary['auto_accepted']}")
    print(f" Flagged for review: {summary['flagged']}")
    print(f" Rejected: {summary['rejected']}")
    return queue

if __name__ == "__main__":
    df = pd.read_csv("data/processed/classification_report.csv")
    build_review_queue(df)