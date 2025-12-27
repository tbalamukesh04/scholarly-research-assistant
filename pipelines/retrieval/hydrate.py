import json
from pathlib import Path

CHUNKS_DIR = Path("data/processed/chunks")

def norm(s: str):
    return (s or "").strip().lower()
def attach_text(retrieval_output: dict) -> dict:
    cache = {}
    
    for r in  retrieval_output["results"]:
        paper_id = r["paper_id"]
        
        if paper_id not in cache:
            chunk_path = CHUNKS_DIR/ f"{paper_id}.json"
            if not chunk_path.exists():
                r["text"] = None
                continue
                
            with chunk_path.open("r", encoding="utf-8") as f:
                cache[paper_id] = json.load(f)
                
        doc = cache[paper_id]
        
        found = False
        for sec in doc.get("sections", []):
            if norm(sec["section"]) != norm(r["section"]):
                continue
                
            for ch in sec.get("chunks", []):
                if ch["chunk_id"] == r["chunk_id"]:
                    r["text"] = ch["text"]
                    found = True
                    break
                    
            if found:
                break
                
        if not found:
            r["text"] = None
            
    return retrieval_output
  