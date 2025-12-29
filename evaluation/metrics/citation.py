import time
from typing import Dict, List

from pipelines.rag.answer import answer
from pipelines.retrieval.search import Retriever


def extract_cited_papers(citations: List[str]) -> List[str]:
    '''
    Extracts the cited papers from a list of citations.
    Args:
        citations: List of citations.
    Returns:
        List of cited papers.
    '''
    papers = set()
    for c in citations:
        #format=> paper_id:section:chunk_id
        papers.add(c.split(":")[0])
    return list(papers)

def evaluate_citations(queries: List[Dict], retriever: Retriever, cache: Dict) -> Dict:
    '''
    Evaluates the citation performance of a given retriever.
    Args:
        queries: List of queries.
        retriever: Retriever to evaluate.
        cache: Cache to store results.
    Returns:
        Evaluation metrics.
    '''
    precisions = []
    recalls = []
    per_query = {}

    for q in queries:
        if q["should_refuse"]:
            continue

        cache_key = f"{q['id']}::strict"
        if cache_key in cache:
            out = cache[cache_key]
        else:
            out = answer(q["query"], mode="strict", retriever=retriever)
            cache[cache_key] = out
            time.sleep(3)

        citations = out.get("citations", [])

        cited_papers = extract_cited_papers(citations)
        relevant = set(q["relevant_papers"])

        if not citations:
            p = 0.0
            r = 0.0

        else:
            correct = sum(1 for p in cited_papers if p in relevant)
            p = correct / len(cited_papers)
            r = correct / len(relevant)

        precisions.append(p)
        recalls.append(r)

        per_query[q["id"]] = {
            "citation_precision": p,
            "recall_precision": r,
            "cited_papers": cited_papers,
            "relevant_papers": list(relevant),
        }

    return {
        "citation_precision_mean": sum(precisions) / len(precisions),
        "citation_recall_mean": sum(recalls) / len(recalls),
        "per_query": per_query,
    }
