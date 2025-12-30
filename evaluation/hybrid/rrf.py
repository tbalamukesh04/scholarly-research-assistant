from collections import defaultdict

def reciprocal_rank_fusion(rankings, k=60):
    '''Reciprocal Rank Fusion (RRF) algorithm for combining multiple rankings.
    
    Args:
        rankings (List[List[Dict]]): A list of rankings, where each ranking is a list of search result items.
        k (int): The maximum rank to consider for scoring.
        
    Returns:
        List[Dict]: A list of fused search result items sorted by score in descending order.
    '''
    scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            key = f"{item['paper_id']}::{item['chunk_id']}"
            scores[key] += 1.0/(k+rank+1)
            assert "paper_id" in item and "chunk_id" in item
            
    fused = []
    for key, score in scores.items():
        paper_id, chunk_id = key.split("::", 1)
        fused.append({
            "paper_id": paper_id, 
            "chunk_id" : chunk_id, 
            "score": score
        })
    return sorted(fused, key = lambda x: x["score"], reverse=True)
    
    