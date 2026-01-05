import json
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

CHUNKS_DIR = Path("data/processed/chunks")


def load_corpus():
    """
    Load the corpus of documents and their metadata.

    Returns:
        tuple: A tuple containing the list of documents and their metadata.
    """
    documents = []
    doc_meta = []

    for path in CHUNKS_DIR.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        paper_id = doc["paper_id"]
        for sec in doc.get("sections", []):
            for ch in sec.get("chunks", []):
                text = ch.get("text")
                if not text:
                    continue

                documents.append(text.lower().split())
                doc_meta.append(
                    {
                        "paper_id": paper_id,
                        "section": sec["section"],
                        "chunk_id": ch["chunk_id"],
                    }
                )
    assert len(documents) > 0, "BM Corpus is Empty"
    return documents, doc_meta


class BM25Retriever:
    def __init__(self):
        corpus, meta = load_corpus()
        self.bm25 = BM25Okapi(corpus)
        self.meta = meta

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform a BM25 search on the corpus.

        Args:
            query (str): The search query.
            k (int): The number of results to return.

        Returns:
            list: A list of normalized results.
        """
        # print(query)
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        results = []

        for idx, score in ranked:
            meta = self.meta[idx]

            results.append(
                {
                    "paper_id": meta["paper_id"],
                    "chunk_id": meta["chunk_id"],
                    "score": float(score),
                }
            )

        return results
