from typing import List, Dict, Any, Optional
from evaluation.utils1 import recall_at_k

class ConfidenceScorer:
    def __init__(self, recall_weight: float = 0.5, alignment_weight: float = 0.5):
        self.recall_weight = recall_weight
        self.alignment_weight = alignment_weight
        
    def _compute_alignment_score(self, alignment_details: List[Dict[str,Any]])-> float:
        if not alignment_details:
            return 0.0
            
        supported_count = sum(1 for item in alignment_details if item.get("supported") is True)
        return supported_count / len(alignment_details)
        
    def calculate(
        self, 
        alignment_details: List[Dict[str, Any]], 
        retrieved_ids: List[str], 
        relevant_papers: List[str], 
        k: int = 10
    ) -> Dict[str, float]:
        
        align_score = self._compute_alignment_score(alignment_details)
        
        recall_score = recall_at_k(retrieved_ids, relevant_papers, k)
        
        total_weight = self.recall_weight + self.alignment_weight
        if total_weight == 0:
            composite_score = 0.0
            
        else:
            w_r = self.recall_weight / total_weight
            w_a = self.alignment_weight / total_weight
            composite_score = (w_r*recall_score) + (w_a * align_score)
            
        return {
            "confidence_score": round(composite_score, 4), 
            "alignment_score": round(align_score, 4), 
            "recall_score": round(recall_score, 4)
        }