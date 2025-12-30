def normalize_result(item):
    '''Normalize a search result item.
    
    Args:
        item (Dict): The search result item.
        
    Returns:
        Dict: The normalized search result item.
    '''
    if "metadata" in item:
        return {
            "paper_id": item["metadata"]["paper_id"], 
            "chunk_id": item["metadata"]["chunk_id"], 
            "score": float(item.get("score", 0.0))
        }
    
    return {
        "paper_id": item["paper_id"], 
        "chunk_id": item["chunk_id"], 
        "score": float(item.get("score", 0.0))
    }