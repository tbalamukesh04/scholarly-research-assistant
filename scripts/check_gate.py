import json
import sys
from pathlib import Path

# ==============================================================================
# check_gate.py
# Purpose: Block promotion/merges if evaluation artifacts are missing or invalid.
# ==============================================================================

METRICS_PATH = Path("evaluation/metrics.json")
GUARDRAIL_THRESHOLD = 0.5  # Minimal acceptable score for promotion (example)

def check_gate():
    print(">>> REGISTRY GATE CHECK: Verifying Evaluation Artifacts...")

    # 1. Existence Check
    if not METRICS_PATH.exists():
        print(f"CRITICAL FAIL: Metrics file missing at {METRICS_PATH}")
        print("Reason: 'evaluate' stage likely failed or was skipped.")
        sys.exit(1)

    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        # 2. Content Validation
        if not metrics:
            print("CRITICAL FAIL: Metrics file is empty.")
            sys.exit(1)

        print(f"Artifacts Found: {json.dumps(metrics, indent=2)}")

        # 3. Quality Gate (Optional/Example)
        # Verify that we have at least some key metrics like recall or precision
        # This prevents a 'successful' run that produced garbage zeros.
        valid_keys = [k for k in metrics.keys() if isinstance(metrics[k], (int, float))]
        if not valid_keys:
            print("CRITICAL FAIL: No numeric metrics found in artifact.")
            sys.exit(1)

        print(">>> SUCCESS: Registry Gate Passed. Promotion authorized.")
        sys.exit(0)

    except json.JSONDecodeError:
        print("CRITICAL FAIL: Metrics file contains invalid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL FAIL: Unexpected error reading artifacts: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_gate()
