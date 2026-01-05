import json

from evaluation.eval_mlf_citation import evaluate_citation
from evaluation.eval_mlf_retrieval import evaluate_retrieval
from evaluation.utils1 import load_queries


def main():
    """
    Runs the MLflow evaluation pipeline.
    """
    print("Loading evaluation queries...")
    queries = load_queries()

    print("Running retrieval evaluation...")
    for retriever_type in ["dense", "hybrid"]:
        print(f"--- Running for retriever_type: {retriever_type} ---")
        retrieval_metrics, _ = evaluate_retrieval(
            queries=queries, retriever_type=retriever_type, k=10
        )
        print(f"Retrieval Metrics ({retriever_type}):")
        print(json.dumps(retrieval_metrics, indent=2))
        print(f"Results saved in evaluation/results/retrieval_{retriever_type}.csv")

    print("\nRunning citation evaluation...")

    citation_metrics, _ = evaluate_citation(
        queries=queries
    )
    print("Citation Metrics:")
    print(json.dumps(citation_metrics, indent=2))
    print("Results saved in evaluation/results/rag_v1.csv")


if __name__ == "__main__":
    main()
