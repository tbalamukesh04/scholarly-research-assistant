import time
from typing import Dict, List

from pipelines.rag.answer import answer
from pipelines.retrieval.search import Retriever

def evaluate_refusals(queries: List[Dict], retriever: Retriever, cache: Dict) -> Dict:
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
    per_query = {}

    for q in queries:
        if not q["should_refuse"]:
            continue

        cache_key = f"{q['id']}::strict"
        if cache_key in cache:
            out = cache[cache_key]
        else:
            out = answer(q["query"], mode="strict", retriever=retriever)
            cache[cache_key] = out
            time.sleep(3)

        refused = (
            "cannot answer" in out["answer"].lower()
            or len(out.get("citations", [])) == 0
        )

        is_correct = refused
        correct += int(is_correct)
        total += 1

        per_query[q["id"]] = {"refused": refused, "correct": is_correct}

    accuracy = correct / total if total > 0 else 1.0

    return {"refusal_accuracy": accuracy, "per_query": per_query}
