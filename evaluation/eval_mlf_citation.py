import pandas as pd
import numpy as np
import os
from pipelines.rag.answer import answer 
# Assuming answer() is the entry point for RAG generation

def evaluate_citation(queries):
    """
    Runs RAG generation and evaluates citation quality.
    Returns schema-compliant metrics + artifact path.
    """
    results = []
    
    # Metrics aggregators
    citation_precisions = []
    citation_recalls = [] 
    avg_confidences = []
    refusal_counts = 0
    
    print(f"Evaluating Citation Quality for {len(queries)} queries...")
    
    for q in queries:
        query_text = q["query"]
        relevant_papers = q["relevant_papers"] # Gold standard for citation recall if applicable
        
        # Run RAG
        # We use eval_mode=True to ensure consistent output format
        response = answer(
            query=query_text, 
            eval_mode=True, 
            relevant_papers=relevant_papers
        )
        
        metrics = response.get("metrics", {})
        
        # Extract schema metrics
        # Note: 'answer' pipeline calculates these. We just aggregate.
        # If 'citation_precision' isn't in metrics, we might need to compute it here 
        # or assume 0 if refused.
        
        cp = metrics.get("citation_precision", 0.0) 
        # For recall, we might need external logic or rely on what 'answer' provides.
        # Let's assume 'recall_score' maps to citation_recall for this context.
        cr = metrics.get("recall_score", 0.0)
        conf = metrics.get("confidence_score", 0.0)
        refused = metrics.get("refusal_triggered", 0.0)
        
        citation_precisions.append(cp)
        citation_recalls.append(cr)
        avg_confidences.append(conf)
        refusal_counts += refused
        
        results.append({
            "query": query_text,
            "answer": response.get("answer", ""),
            "citations": str(response.get("citations", [])),
            "citation_precision": cp,
            "citation_recall": cr,
            "confidence": conf,
            "refused": refused
        })

    # 1. Aggregate Metrics (Schema Compliant Names)
    total = len(queries)
    metrics_summary = {
        "citation_precision": np.mean(citation_precisions),
        "citation_recall": np.mean(citation_recalls),
        "avg_confidence": np.mean(avg_confidences),
        "refusal_rate": refusal_counts / total if total > 0 else 0.0,
        "answer_accuracy": 0.0 # Placeholder: requires human/LLM-judge eval
    }
    
    # 2. Save Artifact
    os.makedirs("evaluation/results", exist_ok=True)
    artifact_path = "evaluation/results/rag_eval_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(artifact_path, index=False)
    
    return metrics_summary, artifact_path