import pandas as pd
import numpy as np
import os
from evaluation.metrics.retrieval import precision_at_k, recall_at_k, reciprocal_rank
from pipelines.retrieval.search import Retriever

def evaluate_retrieval(queries, retriever_type="dense", k=10):
    """
    Runs retrieval evaluation and returns schema-compliant metrics + artifact path.
    """
    # Initialize Retriever
    retriever = Retriever(top_k=k) # Assuming retriever handles type internally or via config
    # Note: If retriever_type affects initialization, adjust accordingly. 
    # For now, we assume standard Retriever.
    
    results = []
    precisions = []
    recalls = []
    mrrs = []
    
    print(f"Evaluating {len(queries)} queries...")
    
    for q in queries:
        query_text = q["query"]
        relevant_ids = set(q["relevant_papers"])
        
        # Search
        search_res = retriever.search(query_text)
        retrieved_ids = [r["paper_id"] for r in search_res["results"]]
        
        # Calculate Metrics
        p_k = precision_at_k(relevant_ids, retrieved_ids, k=k)
        r_k = recall_at_k(relevant_ids, retrieved_ids, k=k)
        mrr = reciprocal_rank(relevant_ids, retrieved_ids, k=k)
        
        precisions.append(p_k)
        recalls.append(r_k)
        mrrs.append(mrr)
        
        results.append({
            "query": query_text,
            "relevant_ids": str(list(relevant_ids)),
            "retrieved_ids": str(retrieved_ids),
            "precision_at_k": p_k,
            "recall_at_k": r_k,
            "mrr": mrr
        })

    # 1. Aggregate Metrics (Schema Compliant Names)
    metrics = {
        "precision_at_k": np.mean(precisions),
        "recall_at_k": np.mean(recalls),
        "mrr": np.mean(mrrs),
        "num_chunks_retrieved": float(k) # Cast to float for MLflow
    }
    
    # 2. Save Artifact (CSV)
    os.makedirs("evaluation/results", exist_ok=True)
    artifact_path = f"evaluation/results/retrieval_{retriever_type}.csv"
    df = pd.DataFrame(results)
    df.to_csv(artifact_path, index=False)
    
    return metrics, artifact_path