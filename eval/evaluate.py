import json
from statistics import mean

from pipelines.retrieval.search import Retriever

TOP_K = 10

def precision_at_k(retrieved_results, relevant_papers, top_k):
    '''
    Calculate the precision@k of the retrieval model.
    Args:
        retrieved_results (list): List of retrieved results.
        relevant_papers (list): List of relevant paper IDs.
        top_k (int): Number of top results to consider.
        
    Returns:
        float: Precision@k value.
    '''
    if not relevant_papers:
        return None
        
    retrieved_paper_ids = [
        r["paper_id"] for r in retrieved_results[:top_k]
    ]
    
    hits = sum(1 for pid in retrieved_paper_ids if pid in relevant_papers)
    
    return hits / top_k 
    
def evaluate_retrieval():
    '''
    Evaluate the precision@k of the retrieval model.
    Args:
        None
        
    Returns:
        None
    '''
    retriever = Retriever(top_k=TOP_K)
    
    with open("eval/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
      
    scores = []
    
    for q in queries:
        if not q["relevant_papers"]:
            continue
            
        result = retriever.search(q["query"])
        retrieved = result["results"]
        
        p = precision_at_k(retrieved, q["relevant_papers"], TOP_K)
        
        if p is not None:
            scores.append(p)
        
        print("\nQuery:", q["query"])
        print("Relevant Papers:", q["relevant_papers"])
        print("Retrieved Papers:", [r["paper_id"] for r in retrieved])
        print("Precision@K:", p)
    
    print("======================SUMMARY======================")
    if scores:
        print("Mean Precision@K:", mean(scores))
        
    else:
        print("No valid queries found.")
        
if __name__ == "__main__":
    evaluate_retrieval()