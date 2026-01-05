from evaluation.mlflow_utils import end_run, log_artifact, start_run, log_metrics
from evaluation.eval_mlf_citation import evaluate_citation
from evaluation.experiments import RAG_PROMPT_V2
from scripts.compute_dataset_hash import compute_dataset_hash
from evaluation.utils1 import load_queries

def run():
    dataset_hash = compute_dataset_hash()
    queries = load_queries()
    
    start_run(RAG_PROMPT_V2, dataset_hash)
    m1, csv1 = evaluate_citation(queries)
    print("M1: ", m1)
    print("csv1: ", csv1)
    log_metrics(m1)
    log_artifact(csv1)
    end_run()
    
if __name__ == "__main__":
    run()
