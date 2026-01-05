from evaluation.mlflow_utils import end_run, start_run, log_metrics
from evaluation.eval_mlf_retrieval import evaluate_retrieval
from evaluation.experiments import RETRIEVAL_DENSE, RETRIEVAL_HYBRID
from scripts.compute_dataset_hash import compute_dataset_hash
from evaluation.utils1 import load_queries

def run():
    dataset_hash = compute_dataset_hash()
    queries = load_queries()
    
    start_run(RETRIEVAL_DENSE, dataset_hash)
    dense_metrics, dense_csv = evaluate_retrieval(queries, retriever_type="dense")
    log_metrics(dense_metrics)
    log_metrics(dense_csv)
    end_run()
    print("Dense metrics: ", dense_metrics)
    print("Dense CSV: ", dense_csv)
    
    start_run(RETRIEVAL_HYBRID, dataset_hash)
    hybrid_metrics, hybrid_csv = evaluate_retrieval(queries, retriever_type="hybrid")
    log_metrics(hybrid_metrics)
    log_metrics(hybrid_csv)
    end_run()
    print("Hybrid Metrics: ", hybrid_metrics)
    print("Hybrid CSV: ", hybrid_csv)
    
if __name__ == "__main__":
    run()