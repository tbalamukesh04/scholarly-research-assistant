import json
import glob
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from utils.helper_functions import get_git_revision_hash, hash_object, hash_text

INPUT_DIR = Path("data/finetuning")
OUTPUT_DIR = Path("data/finetuning/splits")
MANIFEST_PATH = OUTPUT_DIR / "splits_manifest.json"

# Frozen Config
CONFIG = {
    "seed": 42,
    "split_ratios": [0.80, 0.15, 0.05], # Train, Val, Test
    "sources": ["section_qa.jsonl", "partial_qa.jsonl", "refusal_qa.jsonl"]
}

def load_and_hash_jsonl(filename):
    """Loads JSONL and computes a deterministic hash of its content."""
    data = []
    content_hash = []
    
    path = INPUT_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    # Normalize line to ensure hash consistency
                    obj = json.loads(line)
                    data.append(obj)
                    content_hash.append(hash_object(obj))
                except json.JSONDecodeError:
                    continue
    
    # Hash the list of hashes to get a file signature
    return data, hash_object(content_hash)

def save_jsonl_and_hash(data, filepath):
    """Saves data and returns the hash of the file created."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    # Compute hash of the saved data to verify integrity
    # We hash the objects, not the file bytes, to ignore newline diffs across OS
    return hash_object([hash_object(d) for d in data])

def merge_and_shuffle():
    print("---- GENERATING SPLITS WITH LINEAGE ----")
    
    # 1. Load and Hash Inputs (Lineage Tracking)
    input_lineage = {}
    all_examples = []
    
    for source_file in CONFIG["sources"]:
        data, file_hash = load_and_hash_jsonl(source_file)
        input_lineage[source_file] = {
            "count": len(data),
            "hash": file_hash
        }
        all_examples.extend(data)
        print(f"Loaded {source_file}: {len(data)} items (Hash: {file_hash[:8]}...)")

    if not all_examples:
        print("CRITICAL: No examples found.")
        return

    # 2. Deterministic Shuffle
    random.seed(CONFIG["seed"])
    random.shuffle(all_examples)
    
    # 3. Split
    total = len(all_examples)
    r_train, r_val, _ = CONFIG["split_ratios"]
    
    train_end = int(total * r_train)
    val_end = int(total * (r_train + r_val))
    
    splits = {
        "train": all_examples[:train_end],
        "val": all_examples[train_end:val_end],
        "test": all_examples[val_end:]
    }
    
    # 4. Save and Hash Outputs
    output_meta = {}
    for split_name, data in splits.items():
        fname = f"{split_name}.jsonl"
        fpath = OUTPUT_DIR / fname
        file_hash = save_jsonl_and_hash(data, fpath)
        output_meta[split_name] = {
            "file": fname,
            "count": len(data),
            "hash": file_hash
        }

    # 5. Create Manifest (The Artifact Definition)
    manifest = {
        "artifact_type": "dataset_splits",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_revision_hash(),
        "config": CONFIG,
        "input_lineage": input_lineage,
        "outputs": output_meta
    }
    
    # Self-hash
    manifest["artifact_hash"] = hash_object(manifest)
    
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nManifest saved to {MANIFEST_PATH}")
    print(f"Artifact Hash: {manifest['artifact_hash']}")
    print("Lineage secured.")

if __name__ == "__main__":
    merge_and_shuffle()