import json
# import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics.citation import evaluate_citations
from evaluation.metrics.refusal import evaluate_refusals
from evaluation.metrics.retrieval import evaluate_retrieval
from pipelines.retrieval.search import Retriever

shared_retriever = Retriever(top_k=10)

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = Path("evaluation/cache/answers.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

if CACHE_PATH.exists():
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        ANSWER_CACHE = json.load(f)

else:
    ANSWER_CACHE = {}


def main():
    with open("evaluation/queries.json", "r") as f:
        queries = json.load(f)

    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise TypeError(f"Query at index {i} is not a dict")
        for key in ["id", "query", "relevant_papers", "should_refuse"]:
            if key not in q:
                raise KeyError(f"Missing {key} in query {i}")

    try:
        retrieval_metrics = evaluate_retrieval(
            queries=queries, retriever=shared_retriever, k=10
        )

        citation_metrics = evaluate_citations(
            queries=queries, retriever=shared_retriever, cache=ANSWER_CACHE
        )
        refusal_metrics = evaluate_refusals(
            queries=queries, retriever=shared_retriever, cache=ANSWER_CACHE
        )

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval": retrieval_metrics,
            "citation": citation_metrics,
            "refusal": refusal_metrics,
        }

        out_path = (
            RESULTS_DIR
            / f"eval_run_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
        )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(ANSWER_CACHE, f, indent=2)
        raise e

    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(ANSWER_CACHE, f, indent=2)

    print("Evaluation completed.")
    print(json.dumps(results["retrieval"], indent=2))


if __name__ == "__main__":
    main()
