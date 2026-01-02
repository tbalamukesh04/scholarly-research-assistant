# Return Format:
#     {
#         "dataset_name": str, 
#         "dataset_hash": str,
#         "created_at": datetime,
#         'includes': List[str],
#         "excludes": List[str],
#         "chunking": Dict,
#         "embedding":Dict,
#     }

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

MANIFEST_PATH = Path("data/versions/dataset_manifest.json")
CHUNKS_DIR = Path("data/processed/chunks")

def compute_dataset_hash() -> str:
    h = hashlib.sha256()
    
    for chunk_file in sorted(CHUNKS_DIR.glob("*.json")):
        with chunk_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        payload = {
            "paper_id": data["paper_id"], 
            "sections": [
                {
                    "section": sec["section"], 
                    "chunks": [
                        ch["text"] for ch in sec["chunks"]
                    ]
                    }
                    for sec in data["sections"]
            ]
        }
        h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        
    return h.hexdigest()
    
def write_dataset_manifest():
    dataset_hash = compute_dataset_hash()
    
    manifest = {
        "dataset_name": "scholarly-research-assistant", 
        "dataset_hash": f"sha256:{dataset_hash}",
        "created_at": datetime.now(timezone.utc).isoformat(), 
        'includes': [
            "data/processed/chunks/*.json"
        ],
        "excludes": [
        "logs/",
        "evaluation/"
        ], 
        "chunking": {
            "source": "pipelines/processing/extracting_and_chunking_pdfs.py", 
            "size": 64, 
            "overlap": 0
        }, 
        "embedding":{
            "model": "sentence-transformers/all-mpnet-base-v2", 
            "normalized": True
        }
    }
    
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print("Dataset Manifest Written")
    print("Dataset hash: ", manifest["dataset_hash"])
    
if __name__ == "__main__":
    write_dataset_manifest()