import json
from pathlib import Path
from collections import defaultdict

METADATA_DIR = Path("raw/data/metadata")
LINKS_IN = Path("data/processed/dedup_links.json")
LINKS_OUT = Path("data/processed/dedup_links_refined.json")

def load_metadata():
    """
    Loads metadata from Global Metadata Directory
    Returns:
        dict: Metadata
    """
    records = {}
    for p in METADATA_DIR.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            r = json.load(f)
            records[r["paper_id"]] = r
    return records
    
def load_links():
    if not LINKS_IN.exists():
        return {}
        
    with LINKS_IN.open("r", encoding="utf-8") as f:
        return json.load(f)
        
def refine_with_checksums(records, links):
    checksum_index = defaultdict(list)
    
    for pid, r in records.items():
        if r.get("checksum"):
            checksum_index[r["checksum"]].append(pid)
    
    refined = dict(links)
    for checksum, pids in checksum_index.items():
        if len(pids) < 2: 
            continue
        
        primary = pids[0]
        refined[primary] = {
            "aliases": [
                {"paper_id": p, "source": records[p]["source"]} for p in pids 
            ], 
            "match_type": "checksum", 
            "confidence": 1.0
        }
    
    return refined
    
if __name__ == "__main__":
    links = load_links()
    records = load_metadata()
    
    refined = refine_with_checksums(records, links)
    LINKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with LINKS_OUT.open("w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)