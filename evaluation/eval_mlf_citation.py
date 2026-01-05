import csv
from pathlib import Path
from typing import List

from pipelines.rag.answer import answer

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_citation(
    queries: List[dict],
    # prompt_version: str,
):
    """
    Returns:
        metrics: dict
        csv_content: str
    """

    rows = []
    citation_hits = []
    refusal_hits = []

    for q in queries:
        out = answer(
            q["query"],
            mode="strict",
            # prompt_version=prompt_version,
        )

        cited_papers = {c.split(":")[0] for c in out.get("citations", [])}

        gold = set(q["relevant_papers"])

        # Citation precision
        if cited_papers:
            precision = len(cited_papers & gold) / len(cited_papers)
        else:
            precision = 0.0

        citation_hits.append(precision)

        # Refusal accuracy
        refused = len(out.get("citations", [])) == 0
        refusal_hits.append(refused == q["should_refuse"])

        rows.append(
            {
                "query_id": q["id"],
                "citation_precision": precision,
                "refused": refused,
                "should_refuse": q["should_refuse"],
                # "prompt_version": prompt_version,
            }
        )

    metrics = {
        "mean_citation_precision": sum(citation_hits) / len(citation_hits),
        "refusal_accuracy": sum(refusal_hits) / len(refusal_hits),
    }

    csv_path = RESULTS_DIR / "rag_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # csv_content = csv_path.read_text(encoding="utf-8")

    return metrics, csv_path
