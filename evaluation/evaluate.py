import json
import logging
from statistics import mean
from pathlib import Path

from pipelines.retrieval.search import Retriever
from utils.helper_functions import load_yaml

# Define Paths matching DVC dependencies and outputs
QUERIES_PATH = Path("pipelines/evaluation/data/eval_queries.json")
METRICS_PATH = Path("evaluation/metrics.json")
RESULTS_DIR = Path("evaluation/results")

def precision_at_k(retrieved_results, relevant_papers, top_k):
    '''
    Calculate the precision@k of the retrieval model.
    '''
    if not relevant_papers:
        return None
        
    retrieved_paper_ids = [
        r["paper_id"] for r in retrieved_results[:top_k]
    ]
    
    hits = sum(1 for pid in retrieved_paper_ids if pid in relevant_papers)
    
    return hits / top_k 
    
def evaluate_retrieval():
    '''
    Evaluate the precision@k of the retrieval model and save metrics for DVC.
    '''
    # Load configuration
    try:
        params = load_yaml("params.yaml")
        top_k = params["evaluation"]["k"]
    except Exception:
        top_k = 5  # Fallback
        
    retriever = Retriever(top_k=top_k)
    
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Queries file not found at {QUERIES_PATH}")
    
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
      
    scores = []
    details = []
    
    print(f"Starting Evaluation (k={top_k})...")
    
    for q in queries:
        if not q.get("relevant_papers"):
            continue
            
        result = retriever.search(q["query"])
        retrieved = result["results"]
        
        p = precision_at_k(retrieved, q["relevant_papers"], top_k)
        
        if p is not None:
            scores.append(p)
        
        # Collect details for detailed results output
        details.append({
            "query": q["query"],
            "relevant": q["relevant_papers"],
            "retrieved": [r["paper_id"] for r in retrieved],
            "precision": p
        })

        print(f"Query: {q['query'][:40]}... | P@{top_k}: {p}")
    
    mean_p = mean(scores) if scores else 0.0
    
    print("======================SUMMARY======================")
    print(f"Mean Precision@{top_k}: {mean_p}")

    # 1. Write Metrics (Required by DVC 'metrics' field)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"precision_at_k": mean_p}, f, indent=2)
        
    # 2. Write Results (Required by DVC 'outs' field)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
        
if __name__ == "__main__":
    evaluate_retrieval()
