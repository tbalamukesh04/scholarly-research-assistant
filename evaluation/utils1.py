import json
from pathlib import Path

def load_queries():
    # UPDATED: Correct path matching the project structure
    path = Path("pipelines/evaluation/data/eval_queries.json")
    
    if not path.exists():
        raise FileNotFoundError(f"Evaluation queries not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    return queries
    
def precision_at_k(results, relevant_papers, k):
    hits = sum(1 for pid in results if pid in relevant_papers)
    return hits/k
    

def recall_at_k(results, relevant_papers, k):
    hits = sum(1 for pid in results if pid in relevant_papers)
    return 0.0 if len(relevant_papers)==0 else hits/len(relevant_papers)