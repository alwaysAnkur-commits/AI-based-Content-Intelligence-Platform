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

def route_document(doc_id: str, predicted_label: str, confidence: float, batch_id: str) -> dict:
    """Route a single document based on its confidence score."""
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
        "batch_id": batch_id,
        "doc_id": str(doc_id) if pd.notna(doc_id) else "",
        "predicted_label": predicted_label,
        "confidence": confidence,
        "routing_status": status,
        "reason": reason,
        "routed_at": datetime.now().isoformat()
    }

def build_review_queue(classification_df: pd.DataFrame) -> dict:
    """Process classified documents and append routing decisions to review_queue.jsonl."""
    classification_df["doc_id"] = classification_df["doc_id"].fillna("").astype(str)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {"auto_accepted": 0, "flagged": 0, "rejected": 0}
    out_path = Path("data/processed/review_queue.jsonl")

    with open(out_path, "a", encoding="utf-8") as f:
        for _, row in classification_df.iterrows():
            decision = route_document(
                doc_id=row.get("doc_id", ""),
                predicted_label=row.get("predicted_label", ""),
                confidence=float(row.get("confidence", 0.0)),
                batch_id=batch_id
            )
            f.write(json.dumps(decision) + "\n")
            if decision["routing_status"] == "auto_accepted":
                summary["auto_accepted"] += 1
            elif decision["routing_status"] == "flagged_for_review":
                summary["flagged"] += 1
            else:
                summary["rejected"] += 1

    print(f"\nRouting Summary (Batch {batch_id}):")
    print(f" Auto-accepted: {summary['auto_accepted']}")
    print(f" Flagged for review: {summary['flagged']}")
    print(f" Rejected: {summary['rejected']}")
    return summary

if __name__ == "__main__":
    df = pd.read_csv("data/processed/classification_report.csv")
    build_review_queue(df)
