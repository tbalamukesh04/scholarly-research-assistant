import json 
from statistics import mean
from typing_extensions import List, Dict

from evaluation.baselines.bm25 import BM25Retriever

def precision_and_recall_at_k(results: Dict, relevant_papers: List[str], k: int):
    if not relevant_papers:
        return 0.0
        
    retrieved = [r["paper_id"] for r in results[:k]]
    hits = sum(1 for p in retrieved if p in relevant_papers)
    return (hits / k), (hits / len(relevant_papers))
    
def main():
    with open("evaluation/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    retriever = BM25Retriever()
    
    ps, rs = [], []
    
    for q in queries:
        if not q["relevant_papers"]:
            continue
            
        results = retriever.search(q["query"], k=10)
        
        p, r = precision_and_recall_at_k(results, q["relevant_papers"], 10)
        ps.append(p)
        rs.append(r)
        
        print(
            q["id"],
            f"P@10 = {p:.2f}", 
            f"R@10 = {r:.2}",
        )
        
    print("\n================Baseline B25================")
    print("Mean Precision: ", round(mean(ps), 3))
    print("Mean Recall: ", round(mean(rs), 3))
    
if __name__ == "__main__":
    main()