import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sentence_transformers.util.tensor import normalize_embeddings

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into atomic sentences using deterministic regex rules.
    Designed to isolate factual claims for verification.
    """
    if not text:
        return []

    text = re.sub(r'\s+', ' ', text).strip()

    abbrev_2 = r"(?:Mr|Ms|Dr|Jr|Sr|St|vs|cf|eg|ie|ex)"
    
    abbrev_3 = r"(?:Mrs|Fig|Rev|Gov|Col|Cpl|Sgt)"
    
    abbrev_4 = r"(?:Capt|Prof|Dept)"

    pattern = (
        r'(?<!\b' + abbrev_2 + r'\.)'
        r'(?<!\b' + abbrev_3 + r'\.)'
        r'(?<!\b' + abbrev_4 + r'\.)'
        r'(?<!\bet\sal\.)'
        r'(?<!\b[A-Z]\.)'
        r'(?<=[.?!])\s+(?=[A-Z0-9\[])'
    )

    sentences = re.split(pattern, text)

    return [s.strip() for s in sentences if s.strip()]
    
class Attributor:
    def __init__(self, model:SentenceTransformer):
        self.model = model
        
    def verify(self, sentences: List[str], evidence: List[Dict[str, Any]], threshold: float = 0.25)->Dict[str, Any]:
        if not sentences or not evidence:
            return {"attribution_passed": False, "details": [], "reason": "Empty input"}
            
        evidence_texts = [e.get("text", "") or "" for e in evidence]
        
        sent_embs = self.model.encode(sentences, normalize_embeddings=True)
        ev_embs = self.model.encode(evidence_texts, normalize_embeddings=True)
        
        similarity_matrix = np.dot(sent_embs, ev_embs.T)
        
        results = []
        failures = []
        
        for i, sent in enumerate(sentences):
            scores = similarity_matrix[i]
            best_idx = np.argmax(scores)
            max_score = float(scores[best_idx])
            
            supported = max_score >= threshold
            
            record = {
                "sentence": sent, 
                "max_score": max_score, 
                "supported_by_chunk_index": int(best_idx) if supported else None, 
                "supported": supported,
                # Fix: Add the key required by answer.py
                "verification_status": "supported" if supported else "unsupported"
            }
            results.append(record)
            
            if not supported:
                failures.append(f"Sentence {i+1} unsupported (Max Score: {max_score:.2f} < {threshold})")
            
        return {
            "attribution_passed": len(failures) == 0, 
            "details": results, 
            "failures": failures, 
            
        }
        
if __name__ == "__main__":
    sample = (
        "The framework uses a Transformer architecture [1]. "
        "However, regarding efficiency (e.g. memory usage), it lags behind. "
        "Fig. 1 shows the detailed breakdown vs. time. "
        "Smith et al. [2] argue otherwise. "
        "Also check J. K. Rowling [3]."
    )
    results = split_into_sentences(sample)
    for i, s in enumerate(results):
        print(f"{i}: {s}")