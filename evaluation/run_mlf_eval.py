import json
import mlflow
import sys
import os
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).parents[1]))

from evaluation.eval_mlf_citation import evaluate_citation
from evaluation.eval_mlf_retrieval import evaluate_retrieval
from evaluation.utils1 import load_queries
from utils.metadata import PROMPT_VERSION, GUARDRAIL_VERSION, get_index_hash, get_git_commit
from utils.mlflow_handler import MLflowHandler
from utils.mlflow_schema import RunType

DATASET_MANIFEST_PATH = Path("data/versions/dataset_manifest.json")

def get_manifest_dataset_hash() -> str:
    """Reads the dataset hash exactly as it appears in the manifest."""
    if not DATASET_MANIFEST_PATH.exists():
         raise FileNotFoundError(f"Dataset manifest not found at {DATASET_MANIFEST_PATH}")
    with open(DATASET_MANIFEST_PATH, "r") as f:
        data = json.load(f)
    return data.get("dataset_hash")

def main():
    # 1. Setup Experiment
    mlflow.set_experiment("Scholarly-RAG-Evaluation")

    # 2. Resolve Identity for Tags (Must match register.py logic)
    identity_tags = {
        "base_llm": "gemini-2.5-flash-lite", 
        "prompt_version": PROMPT_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
        "dataset_hash": get_manifest_dataset_hash(), 
        "index_hash": get_index_hash(),
        "git_commit": get_git_commit(),
        "run_type": RunType.EVAL.value 
    }

    print(f"📋 Starting Evaluation Run with Identity:\n{json.dumps(identity_tags, indent=2)}")

    # 3. Start the Parent Run (The one register.py will look for)
    with MLflowHandler.start_run(
        run_name="Full_System_Evaluation",
        run_type=RunType.EVAL,
        tags=identity_tags
    ) as run:
        
        print("Loading evaluation queries...")
        queries = load_queries()

        # --- A. Retrieval Evaluation ---
        print("Running retrieval evaluation...")
        for retriever_type in ["dense", "hybrid"]:
            _, r_path = evaluate_retrieval(queries, retriever_type=retriever_type, k=10)
            MLflowHandler.log_artifact(r_path)

        # --- B. Citation & Guardrail Evaluation ---
        print("\nRunning citation evaluation...")
        c_metrics, c_path = evaluate_citation(queries)
        
        # Log Aggregate Metrics (This is what register.py checks against gates)
        MLflowHandler.log_metrics(RunType.EVAL, c_metrics)
        MLflowHandler.log_artifact(c_path)
        
        print("\n✅ Evaluation Complete. Metrics logged to MLflow.")
        print(json.dumps(c_metrics, indent=2))
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()