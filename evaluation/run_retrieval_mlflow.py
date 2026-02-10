from evaluation.mlflow_utils import end_run, start_run, log_metrics, log_artifact
from evaluation.eval_mlf_retrieval import evaluate_retrieval
from scripts.compute_dataset_hash import compute_dataset_hash
from evaluation.utils1 import load_queries

# Define run names as constants or import them
RETRIEVAL_DENSE = "retrieval_dense_run"
RETRIEVAL_HYBRID = "retrieval_hybrid_run"

def run():
    dataset_hash = compute_dataset_hash()
    queries = load_queries()
    
    # --- Dense Run ---
    start_run(RETRIEVAL_DENSE, dataset_hash)
    dense_metrics, dense_artifact = evaluate_retrieval(queries, retriever_type="dense")
    
    # Log Metrics
    log_metrics(dense_metrics)
    
    # Log Artifact (The CSV file)
    log_artifact(dense_artifact)
    
    end_run()
    
    print("Dense Metrics:", dense_metrics)
    print("Dense Artifact:", dense_artifact)
    
    # --- Hybrid Run ---
    # (Assuming evaluate_retrieval handles 'hybrid' logic internally or via arg)
    start_run(RETRIEVAL_HYBRID, dataset_hash)
    hybrid_metrics, hybrid_artifact = evaluate_retrieval(queries, retriever_type="hybrid")
    
    log_metrics(hybrid_metrics)
    log_artifact(hybrid_artifact)
    
    end_run()
    
    print("Hybrid Metrics:", hybrid_metrics)
    print("Hybrid Artifact:", hybrid_artifact)
    
if __name__ == "__main__":
    run()