import json
from evaluation.hybrid.retriever import HybridRetriever
from evaluation.baselines.bm25 import BM25Retriever
from evaluation.metrics_utils.retrieval import recall_at_k, precision_at_k
from pipelines.retrieval.search import Retriever

from statistics import mean
import mlflow

def main():
    '''Main function to run the hybrid evaluation.'''
    queries = json.load(open("evaluation/queries.json"))
    
    bm_25 = BM25Retriever()
    dense = Retriever()
    hybrid = HybridRetriever(dense, bm_25)
    top_k = 10
    p_scores = []
    r_scores = []
    
    for q in queries:
        results = hybrid.search(q["query"], k=top_k)
        
        p = precision_at_k(results, q["relevant_papers"], k=top_k)
        r = recall_at_k(results, q["relevant_papers"], k=top_k)
        
        p_scores.append(p)
        r_scores.append(r)
        
        
    print("============Hybrid Retrieval============")
    print("Mean Precision@10: ", mean(p_scores))
    print("Mean Recall@10: ", mean(r_scores))
    
    with mlflow.start_run(run_name="retrieval_eval"):
        mlflow.log_metric(f"precision@{top_k}", mean(p_scores))
        mlflow.log_metric(f"recall@{top_k}", mean(r_scores))
    
if __name__ == "__main__":
    main()
