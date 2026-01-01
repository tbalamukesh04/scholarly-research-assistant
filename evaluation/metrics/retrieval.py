from typing import Dict, List

from evaluation.hybrid.retriever import HybridRetriever
from pipelines.retrieval.search import Retriever


def precision_at_k(results: List[Dict], relevant_papers: List[str], k: int) -> float:
    '''
    Calculates the precision at k for a given set of results and relevant papers.
    Args:
        results: List of retrieved papers.
        relevant_papers: List of relevant papers.
        k: Number of retrieved papers to consider.
    Returns:
        Precision at k.
    '''
    if not relevant_papers:
        return 0.0
    retrieved = [r["paper_id"] for r in results[:k]]
    hits = sum(1 for pid in retrieved if pid in relevant_papers)
    return hits / k


def recall_at_k(results: List[Dict], relevant_papers: List[str], k: int) -> float:
    '''
    Calculates the recall at k for a given set of results and relevant papers.
    Args:
        results: List of retrieved papers.
        relevant_papers: List of relevant papers.
        k: Number of retrieved papers to consider.
    Returns:
        Recall at k.
    '''
    if not relevant_papers:
        return 0.0
    retrieved = {r["paper_id"] for r in results[:k]}
    hits = sum(1 for pid in relevant_papers if pid in retrieved)
    return hits / len(relevant_papers)


def evaluate_retrieval(queries: List[Dict], retriever: Retriever|HybridRetriever, k: int = 10) -> Dict:
    '''
    Evaluates the retrieval performance of a given retriever.
    Args:
        queries: List of queries.
        retriever: Retriever to evaluate.
        k: Number of retrieved papers to consider.
    Returns:
        Evaluation metrics.
    '''
    precisions = []
    recalls = []

    per_query = {}

    for q in queries:
        if not q["relevant_papers"]:
            continue

        out = retriever.search(q["query"])
        results = out["results"]

        p = precision_at_k(results, q["relevant_papers"], k)
        r = recall_at_k(results, q["relevant_papers"], k)

        precisions.append(p)
        recalls.append(r)

        per_query[q["id"]] = {
            "precision@k": p,
            "recall@k": r,
            "retrieved_papers": [r["paper_id"] for r in results[:k]],
            "relevant_papers": q["relevant_papers"],
        }

    return {
        "precision@k_mean": sum(precisions) / len(precisions),
        "recall@k_mean": sum(recalls) / len(recalls),
        "per_query": per_query,
    }

