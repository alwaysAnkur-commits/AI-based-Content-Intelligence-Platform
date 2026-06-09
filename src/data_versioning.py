import os, json, shutil
from datetime import datetime
from pathlib import Path

VERSION_INDEX = "data/versions/version_index.json"

def get_version_tag() -> str:
    """Generate a version tag like v_20250601_143022."""
    return f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def snapshot(source_dir: str = "data/processed", tag: str = None) -> str:
    """Take a snapshot of the processed data directory."""
    tag = tag or get_version_tag()
    snapshot_dir = f"data/versions/{tag}"
    
    if Path(source_dir).exists():
        shutil.copytree(source_dir, snapshot_dir)
        print(f"Snapshot created: {snapshot_dir}")
    
    # Update version index
    index = load_index()
    index[tag] = {
        "tag": tag,
        "source": source_dir,
        "snapshot_path": snapshot_dir,
        "created_at": datetime.now().isoformat(),
        "files": os.listdir(snapshot_dir) if Path(snapshot_dir).exists() else []
    }
    save_index(index)
    return tag

def rollback(tag: str, target_dir: str = "data/processed"):
    """Restore a previous snapshot to the processed directory."""
    index = load_index()
    if tag not in index:
        print(f"Version {tag} not found. Available: {list(index.keys())}")
        return
    snapshot_path = index[tag]["snapshot_path"]
    if Path(target_dir).exists():
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_path, target_dir)
    print(f"Rolled back to {tag}")

def list_versions():
    index = load_index()
    for tag, meta in sorted(index.items()):
        print(f"  {tag} — {meta['created_at']} — {len(meta['files'])} files")

def load_index() -> dict:
    Path("data/versions").mkdir(parents=True, exist_ok=True)
    if Path(VERSION_INDEX).exists():
        with open(VERSION_INDEX) as f:
            return json.load(f)
    return {}

def save_index(index: dict):
    with open(VERSION_INDEX, "w") as f:
        json.dump(index, f, indent=2)

if __name__ == "__main__":
    tag = snapshot()
    list_versions()