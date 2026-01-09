from typing import List, Dict, Any

def truncate_unsupported_suffix(alignment_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    first_unsupported_idx = None
    
    for i, detail in enumerate(alignment_details):
        if not detail.get("supported", False):
            first_unsupported_idx = i
            break
            
    if first_unsupported_idx is not None:
        return alignment_details[:first_unsupported_idx]
        
    return alignment_details
    
def reconstruct_final_answer(truncated_details: List[Dict[str, Any]])-> str:
    return " ".join([d["sentence"] for d in truncated_details])
    
def apply_strict_truncation(alignment_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i, detail in enumerate(alignment_details):
        if not detail.get("supported", False):
            return alignment_details[:i]
            
        return alignment_details
    
