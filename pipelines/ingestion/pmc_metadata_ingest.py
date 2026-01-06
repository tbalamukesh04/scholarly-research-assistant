import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path

import requests

from utils.logging import setup_logger, log_event
from utils.helper_functions import load_yaml, compute_paper_id

def get_text_safe(element: Optional[ET.Element]) -> str:
    return element.text.strip() if element is not None and element.text else ""
    
def ingest_pmc_metadata():
    project_cfg = load_yaml("configs/project.yaml")
    ingestion_cfg = load_yaml("configs/ingestion.yaml")
    
    log_dir = project_cfg["paths"]["log_root"]
    metadata_dir = Path(ingestion_cfg["storage"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name="pmc_meta_ingestion", 
        log_dir = log_dir, 
        level = logging.INFO
    )
    
    pmc_cfg = ingestion_cfg["sources"]["pmc"]
    if not pmc_cfg.get("enabled", False):
        log_event(
            logger = logger, 
            level = logging.INFO, 
            message = "PMC Ingestion Disabled"
        )
        return 
    terms: List[str] = pmc_cfg.get("terms", ["artifical intelligence"])        
    max_papers:int = pmc_cfg["max_papers"]
    sleep_time = pmc_cfg["sleep_records"]
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    search_query = "%20OR%20".join(terms)
    full_query = f"({search_query}) AND open access[filter]"
    
    params_search = {
        "db": "pmc", 
        "term": full_query, 
        "retmax": max_papers,
        "usehistory": "y", 
        "sort": "date"
    }
    
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "Starting PMC Metadata Ingestion",
        terms = terms, 
        max_papers = max_papers
    )
    
    try:
        resp = requests.get(f"{base_url}/esearch.fcgi", params=params_search)
        resp.raise_for_status()
        
        root_search = ET.fromstring(resp.content)
        id_list = [x.text for x in root_search.findall(".//Id")]
    
    except Exception as e:
        log_event(logger, logging.ERROR, "PMC Metadata Ingestion Failed", error = str(e))
        return
        
    ingested = 0
    skipped = 0
    
    for pmc_uid in id_list:
        source_id = f"PMC{pmc_uid}"
        paper_id = compute_paper_id("pmc", source_id)
        metadata_path = metadata_dir / f"{paper_id}.json"
        
        if metadata_path.exists():
            skipped += 1
            continue
            
        try:
            params_fetch = {"db": "pmc", "id": pmc_uid, "retmode": "xml"}
            resp_fetch = requests.get(f"{base_url}/efetch.fcgi", params=params_fetch)
            resp_fetch.raise_for_status()
            
            root_fetch = ET.fromstring(resp_fetch.content)
            article = root_fetch.find(".//article")
            
            if article is None:
                continue
                
            front = article.find(".//front/article-meta")
            
            
            title_group = front.find("title-group/article-title")
            title = "".join(title_group.itertext()).strip() if title_group is not None else "No Title"

            
            doi_elem = front.find("article-id[@pub-id-type='doi']")
            doi = doi_elem.text if doi_elem is not None else None

            
            authors = []
            for contrib in front.findall(".//contrib-group/contrib[@contrib-type='author']"):
                name = contrib.find("name")
                if name:
                    surname = get_text_safe(name.find("surname"))
                    given = get_text_safe(name.find("given-names"))
                    authors.append(f"{given} {surname}".strip())

            abstract_elem = front.find("abstract")
            abstract = "".join(abstract_elem.itertext()).strip() if abstract_elem is not None else "No Abstract"

            pub_date = "Unknown"
            date_elem = front.find("pub-date[@pub-type='epub']") or front.find("pub-date")
            if date_elem:
                y = get_text_safe(date_elem.find("year"))
                m = get_text_safe(date_elem.find("month")).zfill(2)
                d = get_text_safe(date_elem.find("day")).zfill(2)
                if y:
                    pub_date = f"{y}-{m}-{d}" if m and d else y

            categories = [k.text for k in front.findall(".//kwd-group/kwd") if k.text]

            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{source_id}/pdf/"

            record = {
                "paper_id": paper_id,
                "source": "pmc",
                "source_id": source_id,
                "doi": doi,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published_date": pub_date,
                "updated_date": None,
                "categories": categories,
                "pdf_url": pdf_url,
                "license": "open-access",
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
                message="Ingested PMC metadata",
                paper_id=paper_id,
                source_id=source_id,
            )

            time.sleep(sleep_time)

        except Exception as e:
            log_event(
                logger=logger,
                level=logging.WARNING,
                message="Failed to parse PMC record",
                source_id=source_id,
                error=str(e)
            )

    log_event(
        logger=logger,
        level=logging.INFO,
        message="PMC metadata Ingestion Complete",
        ingested=ingested,
        skipped=skipped,
    )
    
if __name__ == "__main__":
    ingest_pmc_metadata()