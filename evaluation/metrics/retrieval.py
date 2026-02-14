from typing import Dict, List, Any, Union

def precision_at_k(results: List[Dict], relevant_papers: List[str], k: int) -> float:
    '''Calculates precision at k.'''
    if not relevant_papers:
        return 0.0
    
    candidates = results[:k]
    if not candidates:
        return 0.0

    retrieved_ids = [r["paper_id"] for r in candidates]
    hits = sum(1 for pid in retrieved_ids if pid in relevant_papers)
    return hits / len(candidates)

def recall_at_k(results: List[Dict], relevant_papers: List[str], k: int) -> float:
    '''Calculates recall at k.'''
    if not relevant_papers:
        return 0.0
    
    candidates = results[:k]
    retrieved_ids = {r["paper_id"] for r in candidates}
    hits = sum(1 for pid in relevant_papers if pid in retrieved_ids)
    return hits / len(relevant_papers)

def reciprocal_rank(results: List[Dict], relevant_papers: List[str]) -> float:
    '''Calculates MRR.'''
    if not relevant_papers:
        return 0.0
        
    for i, res in enumerate(results):
        if res["paper_id"] in relevant_papers:
            return 1.0 / (i + 1)
            
    return 0.0