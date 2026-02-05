import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from utils.helper_functions import get_git_revision_hash, hash_object

DATASET_MANIFEST = Path("data/versions/dataset_manifest.json")
INDEX_DIR = Path("data/indexes")
INDEX_MANIFEST = INDEX_DIR / "index_manifest.json"

FAISS_INDEX_FILE = "faiss.index"
FAISS_META_FILE = "index_meta.json"

# Frozen Config for Retrieval
RETRIEVAL_CONFIG = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "embedding_dim": 768,
    "similarity_metric": "inner_product",
    "normalization": True
}

def write_index_manifest():
    if not DATASET_MANIFEST.exists():
        print(f"CRITICAL: Dataset manifest not found at {DATASET_MANIFEST}")
        return

    with DATASET_MANIFEST.open("r", encoding="utf-8") as f:
        dataset_manifest = json.load(f)

    # 1. Create the Manifest Payload
    manifest = {
        "artifact_type": "faiss_index",
        "dataset_lineage": {
            "dataset_hash": dataset_manifest["dataset_hash"],
            "dataset_created_at": dataset_manifest["created_at"]
        },
        "config": RETRIEVAL_CONFIG,
        "files": {
            "index": FAISS_INDEX_FILE,
            "metadata": FAISS_META_FILE
        },
        "git_commit": get_git_revision_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # 2. Compute Self-Hash (The ID of this specific index build)
    manifest["artifact_hash"] = hash_object(manifest)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Index Manifest Written to {INDEX_MANIFEST}")
    print(f"Artifact Hash: {manifest['artifact_hash']}")
    print(f"Linked to Git Commit: {manifest['git_commit']}")


if __name__ == "__main__":
    write_index_manifest()