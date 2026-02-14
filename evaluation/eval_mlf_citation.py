import pandas as pd
import numpy as np
import os
from pipelines.rag.answer import answer 

def evaluate_citation(queries):
    """
    Runs RAG generation and evaluates citation quality + refusal correctness.
    Returns schema-compliant metrics + artifact path.
    """
    results = []
    
    citation_precisions = []
    avg_confidences = []
    correct_refusal_count = 0
    
    print(f"Evaluating Citation Quality for {len(queries)} queries...")
    
    for q in queries:
        query_text = q["query"]
        relevant_papers = q.get("relevant_papers", [])
        should_refuse = q.get("should_refuse", False)
        
        # Run RAG
        response = answer(
            query=query_text, 
            eval_mode=True, 
            relevant_papers=relevant_papers
        )
        
        metrics = response.get("metrics", {})
        
        # Extract Metrics
        cp = metrics.get("citation_precision", 0.0)
        conf = metrics.get("confidence_score", 0.0)
        refusal_triggered = metrics.get("refusal_triggered", 0.0) == 1.0
        
        # Logic: Refusal Accuracy
        # Correct if: (Refused AND Should Refuse) OR (Answered AND Should Answer)
        if refusal_triggered == should_refuse:
            correct_refusal_count += 1

        # Only count precision/confidence for non-refused answers (usually)
        # But for MLflow aggregation, we log what we get.
        citation_precisions.append(cp)
        avg_confidences.append(conf)
        
        results.append({
            "query": query_text,
            "answer": response.get("answer", ""),
            "should_refuse": should_refuse,
            "refused": refusal_triggered,
            "citation_precision": cp,
            "confidence": conf
        })

    # Aggregate
    total = len(queries)
    metrics_summary = {
        "citation_precision": np.mean(citation_precisions) if total > 0 else 0.0,
        "avg_confidence": np.mean(avg_confidences) if total > 0 else 0.0,
        "refusal_accuracy": correct_refusal_count / total if total > 0 else 0.0,
        
        # Required schema fillers
        "citation_recall": 0.0, 
        "refusal_rate": sum(1 for r in results if r["refused"]) / total if total > 0 else 0.0,
        "answer_accuracy": 0.0 
    }
    
    # Save Artifact
    os.makedirs("evaluation/results", exist_ok=True)
    artifact_path = "evaluation/results/rag_eval_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(artifact_path, index=False)
    
    return metrics_summary, artifact_path