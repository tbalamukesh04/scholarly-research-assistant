import pandas as pd
import numpy as np
import os
from evaluation.metrics.retrieval import precision_at_k, recall_at_k, reciprocal_rank
from pipelines.retrieval.search import Retriever

def evaluate_retrieval(queries, retriever_type="dense", k=10):
    """
    Runs retrieval evaluation and returns schema-compliant metrics + artifact path.
    """
    retriever = Retriever(top_k=k)
    
    results = []
    precisions = []
    recalls = []
    mrrs = []
    
    print(f"Evaluating {len(queries)} queries...")
    
    for q in queries:
        query_text = q["query"]
        # Convert to list for compatibility
        relevant_ids = list(set(q["relevant_papers"]))
        
        # Search
        search_res = retriever.search(query_text)
        retrieved_items = search_res["results"]
        retrieved_ids = [r["paper_id"] for r in retrieved_items]
        
        # Calculate Metrics
        # Passing 'retrieved_items' (List[Dict]) not 'retrieved_ids'
        p_k = precision_at_k(retrieved_items, relevant_ids, k=k)
        r_k = recall_at_k(retrieved_items, relevant_ids, k=k)
        
        # MRR (Slice input to k)
        mrr = reciprocal_rank(retrieved_items[:k], relevant_ids)
        
        precisions.append(p_k)
        recalls.append(r_k)
        mrrs.append(mrr)
        
        results.append({
            "query": query_text,
            "relevant_ids": str(relevant_ids),
            "retrieved_ids": str(retrieved_ids),
            "precision_at_k": p_k,
            "recall_at_k": r_k,
            "mrr": mrr
        })

    # 1. Aggregate Metrics
    metrics = {
        "precision_at_k": np.mean(precisions) if precisions else 0.0,
        "recall_at_k": np.mean(recalls) if recalls else 0.0,
        "mrr": np.mean(mrrs) if mrrs else 0.0,
        "num_chunks_retrieved": float(k)
    }
    
    # 2. Save Artifact
    os.makedirs("evaluation/results", exist_ok=True)
    artifact_path = f"evaluation/results/retrieval_{retriever_type}.csv"
    df = pd.DataFrame(results)
    df.to_csv(artifact_path, index=False)
    
    return metrics, artifact_path