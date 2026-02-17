import sys
import json
import subprocess
from pathlib import Path

# ANSI Colors for Windows Terminal
GREEN = '\033[92m'
RED = '\033[91m'
NC = '\033[0m'

DATASET_META = Path("data/versions/dataset_manifest.json")
INDEX_MANIFEST = Path("data/processed/faiss/index_manifest.json")

def print_status(msg, status="INFO"):
    if status == "PASS":
        print(f"{GREEN}[PASS] {msg}{NC}")
    elif status == "FAIL":
        print(f"{RED}[FAIL] {msg}{NC}")
    else:
        print(f"[INFO] {msg}")

def verify_lineage():
    print_status("Verifying Artifact Lineage...")
    
    # Check Dataset Metadata
    if not DATASET_META.exists():
        print_status(f"Dataset manifest missing: {DATASET_META}", "FAIL")
        return False
    
    with open(DATASET_META, "r") as f:
        d_meta = json.load(f)
        if "dataset_hash" not in d_meta:
             print_status("dataset_hash missing in manifest", "FAIL")
             return False
    
    # Check Index Manifest
    if not INDEX_MANIFEST.exists():
        print_status(f"Index manifest missing: {INDEX_MANIFEST}", "FAIL")
        return False

    with open(INDEX_MANIFEST, "r") as f:
        i_meta = json.load(f)
        if "artifact_hash" not in i_meta:
            print_status("artifact_hash missing in index manifest", "FAIL")
            return False

    print_status("Lineage hashes verified.", "PASS")
    return True

def verify_determinism():
    print_status("Verifying Determinism...")
    try:
        # call the existing python check script
        subprocess.check_call([sys.executable, "scripts/check_determinism.py"])
        print_status("Determinism Verified.", "PASS")
        return True
    except subprocess.CalledProcessError:
        print_status("Determinism Check Failed.", "FAIL")
        return False

def main():
    print(">>> STARTING REPRODUCIBILITY VERIFICATION (PYTHON) <<<")
    
    if not verify_lineage():
        sys.exit(1)
        
    if not verify_determinism():
        sys.exit(1)
        
    print(f"\n{GREEN}>>> REPRODUCIBILITY VERIFICATION COMPLETE <<<{NC}")
    sys.exit(0)

if __name__ == "__main__":
    main()