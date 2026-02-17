import json
import subprocess
from pathlib import Path

# Constants for System Identity
PROMPT_VERSION = "v1_strict_scholar"
GUARDRAIL_VERSION = "v1_heuristic_threshold"

# UPDATED: Point to the new location defined in dvc.yaml
INDEX_MANIFEST_PATH = Path("data/processed/faiss/index_manifest.json")

def get_git_commit() -> str:
    """Returns the current short git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            text=True
        ).strip()
    except Exception:
        return "unknown"

def get_index_hash() -> str:
    """Retrieves the artifact hash from the FAISS index manifest."""
    if not INDEX_MANIFEST_PATH.exists():
        # Fallback to check if the file exists in the old location just in case
        old_path = Path("data/indexes/index_manifest.json")
        if old_path.exists():
             with open(old_path, "r") as f:
                data = json.load(f)
                return data.get("artifact_hash", "unknown")
        
        return "unknown"
        
    with open(INDEX_MANIFEST_PATH, "r") as f:
        data = json.load(f)
    return data.get("artifact_hash", "unknown")