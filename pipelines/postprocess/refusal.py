from typing import List, Dict, Any, Tuple, Optional

def check_refusal(
    retrieved_chunks: List[Dict[str, Any]], 
    alignment_details: List[Dict[str, Any]], 
    confidence_score: float, 
    confidence_threshold: float, 
    citation_precision: float = 1.0, 
    precision_threshold: float = 0.0, 
    min_distinct_papers: int = 2
) -> Tuple[bool, str]:
    if not retrieved_chunks:
        return True, "Refusal: No evidence retrieved."
        
    paper_ids = {
        chunk.get("paper_id")
        for chunk in retrieved_chunks
        if chunk.get("paper_id")
    }
    
    if len(paper_ids) < min_distinct_papers:
        return True, f"Refusal: Insufficient source diversity ({len(paper_ids)} < {min_distinct_papers})."
        
    if not alignment_details:
        return True, "Refusal: No valid sentences aligned"
        
    for i, detail in enumerate(alignment_details):
        if not detail.get("supported", False):
            return True, f"Refusal: Detected Unsupported sentence at index {i}"
    
    if confidence_score < confidence_threshold:
        return True, f"Refusal: Low Confidence ({confidence_score:.4f} < {confidence_threshold:.4f})."
    
    if citation_precision < precision_threshold:
        return True, f"Refusal: Low Citation Precision ({citation_precision} < {precision_threshold})."
    
    return False, "Pass"
    