import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import feedparser

from utils.logging import log_event, setup_logger
from utils.helper_functions import load_yaml, compute_paper_id


# ------------------------------------------------------------
# -------------------------Core Logic-------------------------
# ------------------------------------------------------------


def ingest_arxiv_metadata():
    """
    Ingests metadata from arxiv.org and stores it in a local directory.
    """
    project_cfg = load_yaml("configs/project.yaml")
    ingestion_cfg = load_yaml("configs/ingestion.yaml")

    log_dir = project_cfg["paths"]["log_root"]
    metadata_dir = Path(ingestion_cfg["storage"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name="arxiv_ingestion", log_dir=log_dir, level=logging.INFO)

    categories: List[str] = ingestion_cfg["sources"]["arxiv"]["categories"]
    max_papers: int = ingestion_cfg["sources"]["arxiv"]["max_papers"]

    base_url = "http://export.arxiv.org/api/query"

    query = "%20OR%20".join(f"cat:{c}" for c in categories)

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_papers,
        "sort_by": "submittedDate",
        "sortOrder": "descending",
    }

    feed_url = (
        f"{base_url}"
        f"?search_query={params['search_query']}"
        f"&start={params['start']}"
        f"&max_results={params['max_results']}"
        f"&sortBy={params['sort_by']}"
        f"&sortOrder={params['sortOrder']}"
    )

    log_event(
        logger=logger,
        level=logging.INFO,
        message="Starting arXiv metadata ingestion",
        categories=categories,
        max_papers=max_papers,
    )

    feed = feedparser.parse(feed_url)

    ingested = 0
    skipped = 0

    for entry in feed.entries:
        source_id = entry.id.split("/")[-1]
        paper_id = compute_paper_id("arxiv", source_id)

        metadata_path = metadata_dir / f"{paper_id}.json"

        if metadata_path.exists():
            skipped += 1
            continue
                    
        pdf_url = None
        for link in entry.links:
            link_type = str(link.get("type"))
            link_href = link.get("href")
            
            if link_type == "application/pdf" and link_href:
                pdf_url = str(link_href)
                break
                
        record = {
            "paper_id": paper_id,
            "source": "arxiv",
            "source_id": str(source_id),
            "doi": str(entry.get("arxiv_doi")) if entry.get("arxiv_doi") else None, 
            "title": str(entry.title).strip(),
            "authors": [str(a.name) for a in entry.authors],
            "abstract": str(entry.summary).strip(),
            "published_date": str(entry.published[:10]),
            "updated_date": str(entry.updated[:10]) if "updated" in entry else None,
            "categories": [str(t.get("term")) for t in entry.tags],
            "pdf_url": pdf_url, 
            "license": None,
            "checksum": None,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        ingested += 1

        log_event(
            logger=logger,
            level=logging.INFO,
            message="Ingested arXiv metadata",
            paper_id=paper_id,
            source_id=source_id,
        )

        time.sleep(1)

    log_event(
        logger=logger,
        level=logging.INFO,
        message="arXiv metadata Ingestion Complete",
        ingested=ingested,
        skipped=skipped,
    )


if __name__ == "__main__":
    ingest_arxiv_metadata()
