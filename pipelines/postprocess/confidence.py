import numpy as np
import re
from typing import List, Dict, Any

class ConfidenceScorer:
    def __init__(self):
        # Weights for the composite score
        self.w_alignment = 0.4
        self.w_retrieval = 0.3
        self.w_precision = 0.3

    def _normalize_id(self, pid: str) -> str:
        """Normalizes arXiv IDs (strips v1 suffixes, prefixes, etc)."""
        if not pid: return ""
        # 1. Lowercase and strip
        pid = pid.lower().strip()
        # 2. Remove 'arxiv:' prefix if present
        pid = pid.replace("arxiv:", "")
        # 3. Remove version suffix (e.g., v1, v2)
        # Matches v followed by digits at the end of string
        pid = re.sub(r"v\d+$", "", pid)
        return pid

    def calculate(
        self, 
        alignment_details: List[Dict], 
        retrieved_ids: List[str], 
        relevant_papers: List[str], 
        k: int
    ) -> Dict[str, float]:
        """
        Calculates a composite confidence score (0.0 - 1.0).
        """
        if not alignment_details:
            return {
                "confidence_score": 0.0, 
                "alignment_score": 0.0, 
                "recall_score": 0.0,
                "citation_precision": 0.0
            }

        # 1. Alignment Score (How much of the answer is supported?)
        supported_count = sum(1 for d in alignment_details if d.get("supported", False))
        total_sentences = len(alignment_details)
        alignment_score = supported_count / total_sentences if total_sentences > 0 else 0.0

        # 2. Retrieval Overlap (Did we find *any* relevant papers?)
        has_relevant = False
        if relevant_papers:
            # Normalize both sides
            # Retrieved IDs might be 'paper_id' or 'paper_id:chunk'
            norm_retrieved = {self._normalize_id(r.split(":")[0]) for r in retrieved_ids}
            norm_relevant = {self._normalize_id(p) for p in relevant_papers}
            
            if not norm_retrieved.isdisjoint(norm_relevant):
                has_relevant = True
        
        recall_score = 1.0 if has_relevant else 0.0 

        # 3. Citation Precision (Of the papers we used, how many were relevant?)
        used_indices = {d["supported_by_chunk_index"] for d in alignment_details if d["supported_by_chunk_index"] is not None}
        
        valuable_citations = 0
        total_citations = len(used_indices)
        
        if total_citations > 0 and relevant_papers:
            norm_relevant = {self._normalize_id(p) for p in relevant_papers}
            for idx in used_indices:
                if idx < len(retrieved_ids):
                    # Robust extraction of paper_id from retrieved ID
                    raw_id = retrieved_ids[idx]
                    # Handle both ':' and '::' separators just in case
                    if "::" in raw_id:
                        paper_id_part = raw_id.split("::")[0]
                    else:
                        paper_id_part = raw_id.split(":")[0]
                        
                    norm_pid = self._normalize_id(paper_id_part)
                    
                    if norm_pid in norm_relevant:
                        valuable_citations += 1
            citation_precision = valuable_citations / total_citations
        else:
            # If relevant papers exist but we cited none -> Precision 0
            # If no relevant papers defined (open ended) -> Precision 1 (benefit of doubt)
            citation_precision = 0.0 if relevant_papers else 1.0

        # 4. Composite Score
        raw_score = (
            (self.w_alignment * alignment_score) +
            (self.w_retrieval * recall_score) +
            (self.w_precision * citation_precision)
        )
        
        return {
            "confidence_score": min(max(raw_score, 0.0), 1.0),
            "alignment_score": alignment_score,
            "recall_score": recall_score,
            "citation_precision": citation_precision
        }