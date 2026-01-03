import json
from datetime import datetime, timezone
from pathlib import Path

DATASET_MANIFEST = Path("data/versions/dataset_manifest.json")
INDEX_DIR = Path("data/indexes")
INDEX_MANIFEST = INDEX_DIR / "index_manifest.json"

FAISS_INDEX_FILE = "faiss.index"
FAISS_META_FILE = "index_meta.json"


def write_index_manifest():
    with DATASET_MANIFEST.open("r", encoding="utf-8") as f:
        dataset_manifest = json.load(f)

    manifest = {
        "index_type": "faiss",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 768,
        "dataset_hash": dataset_manifest["dataset_hash"],
        "built_at": datetime.now(timezone.utc).isoformat(),
        "faiss_index_file": FAISS_INDEX_FILE,
        "faiss_meta_file": FAISS_META_FILE,
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Index manifest written.")
    print("Bound to dataset:", manifest["dataset_hash"])


if __name__ == "__main__":
    write_index_manifest()
