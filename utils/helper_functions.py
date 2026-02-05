import yaml
import hashlib
import json
import subprocess
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# ---------------------Utility Functions----------------------
# ------------------------------------------------------------

def get_git_revision_hash() -> str:
    """
    Retrieves the current git commit hash to tag artifacts.
    """
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown_commit_no_git"

def get_deterministic_json_bytes(data: any) -> bytes:
    """
    Returns bytes of canonical JSON representation with sorted keys and no whitespace.
    Central source of truth for JSON serialization.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

def hash_text(text: str) -> str:
    """
    Canonical text hashing (SHA-256).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def hash_object(data: any) -> str:
    """
    Canonical hashing for any JSON-serializable object.
    """
    return hashlib.sha256(get_deterministic_json_bytes(data)).hexdigest()

def compute_paper_id(source: str, source_id: str) -> str:
    """
    Computes Unique SHA-256 Hash for each file.
    """
    return hash_text(f"{source} :: {source_id}")
    
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
        
def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)