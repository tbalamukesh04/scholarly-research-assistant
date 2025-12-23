import json
import re
from pathlib import Path
from collections import defaultdict


METADATA_DIR = Path("data/raw/metadata")
OUTPUT_PATH = Path("data/processed/dedup_links.json")

def normalize_title(t:str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()

def load_metadata():
    records = {}
    for p in METADATA_DIR.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            r = json.load(f)
            records[r["paper_id"]] = r
    return records
    
def deduplicate(records):
    doi_index = defaultdict(list)
    title_index = defaultdict(list)
    links = {}
    
    for pid, r in records.items():
        if r.get("doi"):
            doi_index[r["doi"]].append(pid)
            
        key = (
            normalize_title(r["title"]), 
            r["authors"][0] if r["authors"] else None, 
            r["published_date"][:4] if r.get("published_date") else None
        )
        
        title_index[key].append(pid)
        
    for doi, pids in doi_index.items():
        if len(pids) > 1:
            primary = pids[0]
            links[primary] = {
                "aliases": [{"paper_id": p, "source": records[p]["source"]} for p in pids],
                "match_type": "doi", 
                "confidence": 1.0, 
            }
            
    for key, pids in title_index.items():
        if len(pids) > 1:
            primary = pids[0]
            if primary in links:
                continue
            links[primary] = {
                "aliases" : [{"paper_id": p, "source": records[p]["source"]} for p in pids],
                "match_type": "heuristic", 
                "confidence": 0.7
            }
            
    return links
    
if __name__ == "__main__":
    records = load_metadata()
    links = deduplicate(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)