import json
from evaluation.hybrid.retriever import HybridRetriever
from evaluation.baselines.bm25 import BM25Retriever
from evaluation.metrics_utils.retrieval import recall_at_k, precision_at_k
from pipelines.retrieval.search import Retriever

def main():
    '''Main function to run the hybrid evaluation.'''
    queries = json.load(open("evaluation/queries.json"))
    
    bm_25 = BM25Retriever()
    dense = Retriever()
    hybrid = HybridRetriever(dense, bm_25)
    
    p_scores = []
    r_scores = []
    
    for q in queries:
        results = hybrid.search(q["query"], k=10)
        
        p = precision_at_k(results, q["relevant_papers"], k=10)
        r = recall_at_k(results, q["relevant_papers"], k=10)
        
        p_scores.append(p)
        r_scores.append(r)
        
        
    print("============Hybrid Retrieval============")
    print("Mean Precision@10: ", sum(p_scores) / len(p_scores))
    print("Mean Recall@10: ", sum(r_scores) / len(r_scores))
    
if __name__ == "__main__":
    main()
