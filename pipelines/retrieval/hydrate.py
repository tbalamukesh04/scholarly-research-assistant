import json
from pathlib import Path
import re

CHUNKS_DIR = Path("data/processed/chunks")

def clean_pdf_artifacts(text: str) -> str:
    if not text: return ""

    text = re.sub(r'(\d)\.?(\d+)?([A-Za-z])', r'\1.\2 \3', text)

    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    
def norm(s: str):
    return (s or "").strip().lower()
def attach_text(retrieval_output: dict) -> dict:
    '''
    Attaches the text of the retrieved documents to the retrieval output.
    Args:
        retrieval_output (dict): The retrieval output.
    Returns:
        dict: The retrieval output with the text attached.
    '''
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
                    r["text"] = clean_pdf_artifacts(ch["text"])
                    found = True
                    break
                    
            if found:
                break
                
        if not found:
            r["text"] = None
            
    return retrieval_output
  
