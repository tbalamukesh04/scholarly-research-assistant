import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from utils.helper_functions import get_deterministic_json_bytes

# UPDATED: Path matches new dvc.yaml structure
metadata_PATH = Path("data/versions/dataset_manifest.json") 
CHUNKS_DIR = Path("data/processed/chunks")

def compute_dataset_hash() -> str:
    h = hashlib.sha256()
    
    if not CHUNKS_DIR.exists():
        print(f"ERROR: Chunks directory not found at {CHUNKS_DIR}")
        sys.exit(1)

    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        print(f"WARNING: No chunk files found in {CHUNKS_DIR}")

    for chunk_file in chunk_files:
        with chunk_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        payload = {
            "paper_id": data["paper_id"],
            "sections": [
                {
                    "section": sec["section"],
                    "chunks": [ch["text"] for ch in sec["chunks"]],
                }
                for sec in data["sections"]
            ],
        }
        h.update(get_deterministic_json_bytes(payload))

    return h.hexdigest()

def write_dataset_metadata():
    print(f"Computing hash from: {CHUNKS_DIR}")
    dataset_hash = compute_dataset_hash()

    metadata = {
        "dataset_name": "scholarly-research-assistant",
        "dataset_hash": f"sha256:{dataset_hash}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "includes": ["data/processed/chunks/*.json"],
        "excludes": ["logs/", "evaluation/"],
        "chunking": {
            "source": "pipelines/processing/extracting_and_chunking_pdfs.py",
            "size": 64,
            "overlap": 0,
        },
        "embedding": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "normalized": True,
        },
    }

    metadata_PATH.parent.mkdir(parents=True, exist_ok=True)
    with metadata_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Manifest Written: {metadata_PATH}")
    print(f"   Hash: {metadata['dataset_hash']}")

if __name__ == "__main__":
    write_dataset_metadata()