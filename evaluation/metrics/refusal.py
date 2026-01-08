import time
from typing import Dict, List

from evaluation.hybrid.retriever import HybridRetriever
from pipelines.rag.answer import answer
from pipelines.retrieval.search import Retriever

def evaluate_refusals(queries: List[Dict], retriever: Retriever|HybridRetriever, cache: Dict, confidence_threshold: float = 0.0) -> Dict:
    '''
    Evaluates the refusal performance of a given retriever.
    Args:
        queries: List of queries.
        retriever: Retriever to evaluate.
        cache: Cache to store results.
    Returns:
        Evaluation metrics.
    '''
    correct = 0
    total = 0
    confidence_refusal_count = 0
    per_query = {}

    for q in queries:
        if not q["should_refuse"]:
            continue

        cache_key = f"{q['id']}::strict"
        
        use_cache = False
        if cache_key in cache:
            cached_metrics = cache[cache_key].get("metrics", {})
            if "confidence_score" in cached_metrics:
                use_cache = True
        
        if use_cache:
            out = cache[cache_key]
        else:
            out = answer(
            q["query"],
            mode="strict",
            retriever=retriever,
            eval_mode=True,
            relevant_papers=q.get("relevant_papers", []),
            confidence_threshold = confidence_threshold 
            )
            cache[cache_key] = out
            time.sleep(3)
        
        ans_text = out.get("answer", "") or ""
        refused = (
            "cannot answer" in ans_text.lower()
            or len(out.get("citations", [])) == 0
            or out.get("metrics", {}).get("refused", False)
        )
        
        refusal_reason = out.get("metrics", {}).get("refusal_reason", "")
        if refused and "Low Confidence" in refusal_reason:
            confidence_refusal_count += 1
            
        is_correct = refused
        correct += int(is_correct)
        total += 1

        per_query[q["id"]] = {
            "refused": refused,
            "correct": is_correct,
            "confidence": out.get("metrics", {}).get("confidence_score", 0.0), 
           "refusal_reason": refusal_reason 
        }

    accuracy = correct / total if total > 0 else 1.0

    return {
        "refusal_accuracy": accuracy,
        "confidence_refusals": confidence_refusal_count,
        "per_query": per_query,
        
    }
