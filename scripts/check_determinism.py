import sys
import json
import hashlib
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from pipelines.rag.answer import answer

# Fixed Query per Day 41 Instructions
QUERY = "What problem does C2LLM aim to solve?"

def hash_response(res):
    '''
    Compute a hash of the structural elements that must remain constant.
    We ignore latency/timing metrics and focus on content identity.
    '''
    payload = {
        # Check Retrieved Chunk IDs (Sequence & Content matters)
        "citations": sorted(res.get("citations", [])),
        
        # Check Refusal State (Must be identical)
        "refusal": res.get("metrics", {}).get("refusal_triggered", 0.0),
        
        # Check Confidence Score (Rounded to 4 decimals to avoid float drift)
        "confidence": round(res.get("metrics", {}).get("confidence_score", 0.0), 4),
        
        # Check Alignment/Factuality Score
        "alignment": round(res.get("metrics", {}).get("alignment_score", 0.0), 4)
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

def main():
    print(f"Running Determinism Check for query: '{QUERY}'")

    # Run 1
    print("  > Execution 1...")
    res1 = answer(QUERY, mode="strict")
    hash1, payload1 = hash_response(res1)

    # Run 2
    print("  > Execution 2...")
    res2 = answer(QUERY, mode="strict")
    hash2, payload2 = hash_response(res2)

    # Comparison
    if hash1 != hash2:
        print("\nFAIL: Non-deterministic output detected!")
        print(f"Run 1 Hash: {hash1}")
        print(f"Run 2 Hash: {hash2}")
        print("-" * 30)
        print(f"Run 1 Details: {json.dumps(payload1, indent=2)}")
        print(f"Run 2 Details: {json.dumps(payload2, indent=2)}")
        sys.exit(1)

    print(f"SUCCESS: Outputs are deterministic. Hash: {hash1}")
    sys.exit(0)

if __name__ == "__main__":
    main()
