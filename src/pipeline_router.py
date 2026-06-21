import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_routing_config(config_path: str = "src/routing_config.json") -> dict:
    """Load routing rules from config file."""
    with open(config_path) as f:
        return json.load(f)

def route_to_pipeline(doc_id: str, label: str, config: dict, batch_id: str) -> dict:
    """Map a classified label to the correct downstream pipeline."""
    rules = config.get("routing_rules", {})
    if label in rules:
        route = rules[label]
    else:
        route = {
            "pipeline": config.get("fallback_pipeline", "general_nlp"),
            "description": "Fallback: unknown document type",
            "next_steps": ["general_nlp"]
        }
    return {
        "batch_id": batch_id,
        "doc_id": str(doc_id) if doc_id else "",
        "classified_as": label,
        "assigned_pipeline": route["pipeline"],
        "next_steps": route["next_steps"],
        "description": route["description"],
        "routed_at": datetime.now().isoformat()
    }

def build_routing_log(accepted_df: pd.DataFrame) -> pd.DataFrame:
    """Route all auto-accepted documents and append results to routing_log.csv."""
    config = load_routing_config()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    routing_log = []
    pipeline_counts = {}

    for _, row in accepted_df.iterrows():
        route = route_to_pipeline(
            doc_id=row.get("doc_id", ""),
            label=row.get("predicted_label", ""),
            config=config,
            batch_id=batch_id
        )
        routing_log.append(route)
        pipe = route["assigned_pipeline"]
        pipeline_counts[pipe] = pipeline_counts.get(pipe, 0) + 1

    log_df = pd.DataFrame(routing_log)

    out_path = Path("data/processed/routing_log.csv")
    log_df.to_csv(
        out_path,
        mode="a",  # append mode
        header=not out_path.exists(),  # write header only if file doesn't exist
        index=False
    )

    print(f"\nPipeline Distribution (Batch {batch_id}):")
    for pipe, count in sorted(pipeline_counts.items(), key=lambda x: -x[1]):
        print(f"  {pipe}: {count} documents")

    return log_df

if __name__ == "__main__":
    # Load auto-accepted docs from review queue (JSONL or JSON depending on your setup)
    queue_path = Path("data/processed/review_queue.jsonl")
    if queue_path.exists():
        # JSONL version: read line by line
        records = [json.loads(line) for line in open(queue_path, "r", encoding="utf-8")]
        accepted_records = [r for r in records if r.get("routing_status") == "auto_accepted"]
        accepted_df = pd.DataFrame(accepted_records)
    else:
        # Fallback to old JSON snapshot
        with open("data/processed/review_queue.json") as f:
            queue = json.load(f)
        accepted_df = pd.DataFrame(queue.get("auto_accepted", []))

    build_routing_log(accepted_df)
