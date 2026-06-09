import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_routing_config(config_path: str = "src/routing_config.json") -> dict:
    with open(config_path) as f:
        return json.load(f)

def route_to_pipeline(doc_id: str, label: str, config: dict) -> dict:
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
        "doc_id": doc_id,
        "classified_as": label,
        "assigned_pipeline": route["pipeline"],
        "next_steps": route["next_steps"],
        "description": route["description"],
        "routed_at": datetime.now().isoformat()
    }

def build_routing_log(accepted_df: pd.DataFrame) -> pd.DataFrame:
    """Route all auto-accepted documents to their pipelines."""
    config = load_routing_config()
    routing_log = []
    pipeline_counts = {}
    
    for _, row in accepted_df.iterrows():
        route = route_to_pipeline(
            doc_id=row.get("doc_id", ""),
            label=row.get("predicted_label", ""),
            config=config
        )
        routing_log.append(route)
        pipe = route["assigned_pipeline"]
        pipeline_counts[pipe] = pipeline_counts.get(pipe, 0) + 1
    
    log_df = pd.DataFrame(routing_log)
    log_df.to_csv("data/processed/routing_log.csv", index=False)
    
    print(f"\nPipeline Distribution:")
    for pipe, count in sorted(pipeline_counts.items(), key=lambda x: -x[1]):
        print(f"  {pipe}: {count} documents")
    
    return log_df

if __name__ == "__main__":
    # Load auto-accepted docs from Task 8 output
    with open("data/processed/review_queue.json") as f:
        queue = json.load(f)
    accepted_df = pd.DataFrame(queue["auto_accepted"])
    build_routing_log(accepted_df)