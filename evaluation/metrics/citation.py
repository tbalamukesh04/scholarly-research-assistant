import time
from typing import Dict, List

from evaluation.hybrid.retriever import HybridRetriever
from pipelines.rag.answer import answer
from pipelines.retrieval.search import Retriever


def extract_cited_papers(citations: List[str]) -> List[str]:
    """
    Extracts the cited papers from a list of citations.
    Args:
        citations: List of citations.
    Returns:
        List of cited papers.
    """
    papers = set()
    for c in citations:
        # format=> paper_id:section:chunk_id
        papers.add(c.split(":")[0])
    return list(papers)


def evaluate_citations(
    queries: List[Dict], retriever: Retriever | HybridRetriever, cache: Dict
) -> Dict:
    """
    Evaluates the citation performance of a given retriever.
    Args:
        queries: List of queries.
        retriever: Retriever to evaluate.
        cache: Cache to store results.
    Returns:
        Evaluation metrics.
    """
    correct = 0
    total = 0

    for q in queries:
        if q["should_refuse"]:
            continue

        cache_key = f"{q['id']}::strict"
        if cache_key in cache:
            out = cache[cache_key]
        else:
            print(q["query"])
            out = answer(q["query"], mode="strict", retriever=retriever)
            cache[cache_key] = out
            time.sleep(3)

        if not out.get("citations"):
            continue

        cited_papers = {c.split(":")[0] for c in out["citations"]}

        if any(p in cited_papers for p in q["relevant_papers"]):
            correct += 1

        total += 1

    return {
        "citation_precision": correct / total if total else 0.0,
        "total_evaluated": total,
    }
