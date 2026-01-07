import re
from typing import List, Dict, Any

class HallucinationChecker:
    def __init__(self):
        # Corrected pattern to capture integers inside brackets: [1], [10]
        self.citation_pattern = re.compile(r'\[(\d+)\]')

    def run_checks(self, answer_text: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates that [N] refers to a valid index in the evidence list (1-based).
        """
        
        # 1. Extract all citation numbers
        found_matches = self.citation_pattern.findall(answer_text)
        
        # Convert to integers (1-based index)
        cited_indices = []
        for m in found_matches:
            try:
                cited_indices.append(int(m))
            except ValueError:
                continue 
            
        errors = []
        valid_indices = []
        
        # 2. Check: Are there any citations at all?
        if not cited_indices:
            errors.append("No citations found in answer.")

        # 3. Check bounds
        num_evidence = len(evidence)
        invalid_indices = []
        
        for idx in cited_indices:
            if 1 <= idx <= num_evidence:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)
        
        if invalid_indices:
            errors.append(f"Citations found that are out of bounds (available 1-{num_evidence}): {invalid_indices}")

        passed = len(errors) == 0

        return {
            "verification_passed": passed,
            "errors": errors,
            "cited_indices": sorted(list(set(valid_indices)))
        }