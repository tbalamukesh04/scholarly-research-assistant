import numpy as np
from typing import List, Dict, Any

class ConfidenceScorer:
    def __init__(self):
        # Weights for the composite score
        self.w_alignment = 0.4
        self.w_retrieval = 0.3
        self.w_precision = 0.3

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
        # 1.0 if all sentences are supported
        supported_count = sum(1 for d in alignment_details if d.get("supported", False))
        total_sentences = len(alignment_details)
        alignment_score = supported_count / total_sentences if total_sentences > 0 else 0.0

        # 2. Retrieval Overlap (Did we find *any* relevant papers?)
        # Simple binary check: Is there any intersection?
        has_relevant = False
        if relevant_papers:
            # Fuzzy match: "paper_id" in relevant vs "paper_id:chunk" in retrieved
            # We strip the chunk suffix to compare
            clean_retrieved = {r.split(":")[0] for r in retrieved_ids}
            clean_relevant = set(relevant_papers)
            if not clean_retrieved.isdisjoint(clean_relevant):
                has_relevant = True
        
        recall_score = 1.0 if has_relevant else 0.0 # Simplified recall for confidence

        # 3. Citation Precision (Of the papers we used, how many were relevant?)
        # We look at the 'supported_by_chunk_index' in alignment details
        used_indices = {d["supported_by_chunk_index"] for d in alignment_details if d["supported_by_chunk_index"] is not None}
        
        valuable_citations = 0
        total_citations = len(used_indices)
        
        if total_citations > 0 and relevant_papers:
            for idx in used_indices:
                if idx < len(retrieved_ids):
                    chunk_id = retrieved_ids[idx]
                    paper_id = chunk_id.split(":")[0]
                    if paper_id in relevant_papers:
                        valuable_citations += 1
            citation_precision = valuable_citations / total_citations
        else:
            # If we didn't cite anything, precision is undefined (or 1.0 if we didn't need to?)
            # Let's be strict: if we didn't cite but produced an answer, verify alignment.
            citation_precision = 0.0 if relevant_papers else 1.0

        # 4. Composite Score
        # Ensure it never exceeds 1.0
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