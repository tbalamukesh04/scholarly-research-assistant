import csv
from pathlib import Path
from typing import Dict, List, Tuple

from evaluation.baselines.bm25 import BM25Retriever
from evaluation.hybrid.retriever import HybridRetriever
from evaluation.utils1 import precision_at_k, recall_at_k
from pipelines.retrieval.search import Retriever

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_retrieval(
    queries: List[dict],
    retriever_type: str,
    k: int = 10,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
        metrics: dict (Aggregate means)
        per_query_metrics: dict (Flattened row-level metrics compatible with mlflow.log_metric)
    """

    if retriever_type == "dense":
        retriever = Retriever()
    else:
        dense = Retriever()
        bm25 = BM25Retriever()
        retriever = HybridRetriever(dense, bm25)

    rows = []
    p_vals, r_vals = [], []
    # print(queries)

    for q in queries:
        # print(q["query"])
        retrieved = retriever.search(q["query"])
        if retriever_type == "dense":
            retrieved = retrieved["results"]
        retrieved_papers = [r["paper_id"] for r in retrieved]

        p = precision_at_k(retrieved_papers, q["relevant_papers"], k)
        r = recall_at_k(retrieved_papers, q["relevant_papers"], k)

        p_vals.append(p)
        r_vals.append(r)

        rows.append(
            {
                "query_id": q["id"],
                "precision@k": p,
                "recall@k": r,
                "retriever": retriever_type,
            }
        )

    # 1. Aggregate Metrics
    metrics = {
        "mean_precision_k": sum(p_vals) / len(p_vals) if p_vals else 0.0,
        "mean_recall_k": sum(r_vals) / len(r_vals) if r_vals else 0.0,
    }

    # 2. Write CSV (Side Effect)
    csv_path = RESULTS_DIR / f"retrieval_{retriever_type}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # 3. Create Flattened Dictionary for MLflow
    # Format: "metric_name_query_id": value
    per_query_metrics = {}
    for row in rows:
        qid = row["query_id"]
        per_query_metrics[f"precision_k_id_{qid}"] = float(row["precision@k"])
        per_query_metrics[f"recall_k_id_{qid}"] = float(row["recall@k"])

    return metrics, per_query_metrics
