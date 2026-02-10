import subprocess
import json
import os
from functools import lru_cache

# Define explicit versions for logic that isn't yet tracked by files
PROMPT_VERSION = "v1_strict_scholar"
GUARDRAIL_VERSION = "v1_heuristic_threshold"

@lru_cache(maxsize=1)
def get_git_commit() -> str:
    """Retrieves the current git commit hash (short)."""
    try:
        # Run git command to get the short hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        # Fallback for environments without git (e.g. docker containers without .git)
        # In Strict Mode, we might want to fail here, but for now we mark it.
        return "dirty_no_git"

@lru_cache(maxsize=1)
def get_index_hash() -> str:
    """Retrieves the FAISS index hash from the manifest."""
    # Assuming the manifest is generated at this path by the ingestion pipeline
    manifest_path = "data/versions/dataset_manifest.json" 
    
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
                # Look for index_hash, or dataset_hash if index isn't separate
                return data.get("index_hash", data.get("dataset_hash", "unknown_index"))
        except Exception:
            return "corrupt_manifest"
            
    return "missing_manifest"