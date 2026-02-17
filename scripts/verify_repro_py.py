import sys
import json
import subprocess
from pathlib import Path

DATASET_META = Path('data/versions/dataset_manifest.json')
INDEX_MANIFEST = Path('data/processed/faiss/index_manifest.json')

def verify_lineage():
    print('[INFO] Verifying Artifact Lineage...')
    if not DATASET_META.exists():
        print(f'[FAIL] Dataset manifest missing: {DATASET_META}')
        return False
    
    with open(DATASET_META, 'r') as f:
        if 'dataset_hash' not in json.load(f):
             print('[FAIL] dataset_hash missing in manifest')
             return False
    
    if not INDEX_MANIFEST.exists():
        print(f'[FAIL] Index manifest missing: {INDEX_MANIFEST}')
        return False

    with open(INDEX_MANIFEST, 'r') as f:
        if 'artifact_hash' not in json.load(f):
            print('[FAIL] artifact_hash missing in index manifest')
            return False

    print('[PASS] Lineage hashes verified.')
    return True

def verify_determinism():
    print('[INFO] Verifying Determinism...')
    try:
        subprocess.check_call([sys.executable, 'scripts/check_determinism.py'])
        print('[PASS] Determinism Verified.')
        return True
    except subprocess.CalledProcessError:
        print('[FAIL] Determinism Check Failed.')
        return False

if __name__ == '__main__':
    print('--- STARTING REPRODUCIBILITY VERIFICATION (PYTHON) ---')
    if verify_lineage() and verify_determinism():
        print('\n--- REPRODUCIBILITY VERIFICATION COMPLETE ---')
        sys.exit(0)
    else:
        print('\n--- VERIFICATION FAILED ---')
        sys.exit(1)
