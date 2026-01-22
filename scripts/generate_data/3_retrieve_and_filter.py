import json
import time
import os
import sys

# Ensure we can import from pipelines
sys.path.append(os.getcwd())

from pipelines.retrieval.search import Retriever
from pipelines.retrieval.hydrate import attach_text

def adapt_for_rag(results, query):
    return {
        "query": query,
        "results": [
            {
                "paper_id": r["paper_id"],
                "chunk_id": r["chunk_id"],
                # Robust section extraction
                "section": r["chunk_id"].split("::")[2].lower() if "::" in r["chunk_id"] else "unknown",
                "text": None,
            }
            for r in results
        ],
    }

def is_garbage(text):
    if not text: return True
    # If >20% of chars are non-printable or weird symbols, it's garbage
    bad_chars = sum(1 for c in text if not c.isprintable())
    if len(text) == 0 or (bad_chars / len(text) > 0.2): return True
    return False

def normalize_section_name(name):
    name = name.lower()
    if "intro" in name: return "introduction"
    if "method" in name: return "methodology"
    if "result" in name: return "results"
    if "discuss" in name: return "discussion"
    if "conclu" in name: return "conclusion"
    if "abstract" in name: return "abstract"
    if "related" in name: return "related work"
    return name

def filter_queries():
    print("------------------------------------------------")
    print("   RUNNING STRICT FILTERING SCRIPT (v2)      ")
    print("------------------------------------------------")
    
    try:
        with open("data_candidates.json", "r") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        print("Error: data_candidates.json not found.")
        return

    retriever = Retriever(top_k=10) 
    validated_data = []

    print(f"Processing {len(candidates)} queries...")

    for i, query in enumerate(candidates):
        try:
            # 1. Identify Target Section from Query
            target_section = None
            query_lower = query.lower()
            
            for sec in ["abstract", "introduction", "related work", "methodology", "results", "discussion", "conclusion"]:
                if sec in query_lower:
                    target_section = normalize_section_name(sec)
                    break
            
            if not target_section:
                continue 

            # 2. Search
            raw = retriever.search(query)
            results = raw.get("results", [])

            # 3. Hydrate
            retrieved_obj = adapt_for_rag(results, query)
            hydrated = attach_text(retrieved_obj)
            evidence = hydrated["results"]

            # 4. STRICT FILTERING
            valid_evidence = []
            for e in evidence:
                # Check A: Garbage
                if is_garbage(e.get("text")):
                    continue
                
                # Check B: Section Match
                chunk_section = normalize_section_name(e["section"])
                
                # Loose matching (e.g., "method" matches "methodology")
                if target_section in chunk_section or chunk_section in target_section:
                    valid_evidence.append(e)

            # 5. Threshold: Must have at least 1 CLEAN, MATCHING chunk
            if len(valid_evidence) < 1:
                continue

            validated_data.append({
                "query": query,
                "evidence": valid_evidence
            })

        except Exception as e:
            print(f"Error processing {query}: {e}")

        if i % 100 == 0:
            print(f"Processed {i} | Kept {len(validated_data)} Strict Matches")

    with open("data_validated.json", "w") as f:
        json.dump(validated_data, f, indent=2)
    
    print(f"DONE. Saved {len(validated_data)} high-quality examples to data_validated.json")

if __name__ == "__main__":
    filter_queries()