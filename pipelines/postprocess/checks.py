import re
from typing import List, Dict, Any

from pipelines.postprocess.align import split_into_sentences

class HallucinationChecker:
    def __init__(self):
        # Corrected pattern to capture integers inside brackets: [1], [10], [1, 2]
        self.citation_pattern = re.compile(r'\[([\d,\s]+)\]')

    def run_checks(self, answer_text: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates that [N] refers to a valid index in the evidence list (1-based).
        """
        sentences = split_into_sentences(answer_text)
        errors = []
        valid_indices = set()
        num_evidence = len(evidence)
        
        if not sentences:
            return {
                "verification_passed": False, 
                "errors": ["Empty Response Received"], 
                "cited_indices":[]
            }
        
        for i, sent in enumerate(sentences):
            found_matches = self.citation_pattern.findall(sent)
        
            if not found_matches:
                continue
                
            for m in found_matches:
                # Split by comma to handle [1, 2] format
                parts = [p.strip() for p in m.split(',')]
                for part in parts:
                    if not part:
                        continue
                    try:
                        idx = int(part)
                    except ValueError:
                        continue 
                        
                    if 1 <= idx <= num_evidence:
                        valid_indices.add(idx)
                    else:
                        errors.append(f"Sentence {i+1} cites out of bounds [{idx}]")
                
        passed = len(errors) == 0
        
        return {
            "verification_passed": passed, 
            "errors": errors, 
            "cited_indices": sorted(list(valid_indices))
        }