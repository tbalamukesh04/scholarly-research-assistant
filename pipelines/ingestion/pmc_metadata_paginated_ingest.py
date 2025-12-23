import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import requests
from xml.etree import ElementTree as ET
from utils.logging import setup_logger, log_event
from utils.helper_functions import load_yaml, compute_paper_id

def ingest_pmc_metadata_paginated():
    project_cfg = load_yaml("configs/project.yaml")
    ingestion_cfg = load_yaml("configs/ingestion.yaml")
    
    log_dir = project_cfg["paths"]["log_root"]
    metadata_dir = Path(ingestion_cfg["storage"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    pmc_cfg = ingestion_cfg["sources"]["pmc"]
    max_records = int(pmc_cfg.get("max_records", 500))
    sleep_records = float(pmc_cfg.get("max_records", 0.5))
    
    logger = setup_logger(
        name = "pmc_paginated_metadata", 
        level = logging.INFO, 
        log_dir = log_dir
    )
    
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
    
    params = {
        "verb": "ListRecords", 
        "metadataPrefix": "oai_dc", 
        "set" : "pmc-open"
    }
    
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    
    ingested = 0
    skipped = 0 
    token: Optional[str] = None
    
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "Starting PMC paginated metadata collection",
        max_records = max_records
    )
    
    while True:
        req_params = {"verb": "ListRecords"} if token else params.copy()
        
        if token:
            req_params["resumptionToken"] = token
            
        response = requests.get(base_url, params=req_params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        
        records = root.findall(".//oai:record", ns)
        
        for rec in records:
            if ingested > max_records:
                log_event(
                    logger = logger, 
                    level = logging.INFO, 
                    message = "PMC Record cap reached",
                    ingested = ingested
                )
                return
            header = rec.find("oai:header", ns)
            if header is None:
                continue
                
            pmc_id = header.findtext("oai:identifier", default=None, namespaces=ns)
            if not pmc_id:
                continue
                
            paper_id = compute_paper_id("pmc", pmc_id)
            meta_path = metadata_dir / f"{paper_id}.json"
            if meta_path.exists():
                skipped += 1
                continue
                
            meta = rec.find("oai:metadata", ns)
            if meta is None:
                continue
                
            dc = meta.find("dc:dc", ns)
            if dc is None:
                continue
                
            title = dc.findtext("dc:title", default="")
            abstract = dc.findtext("dc:description", default="")
            authors = [e.text for e in dc.findall("dc:creator", ns) if e.text]
            date = dc.findtext("dc:date", default=None)
            
            record_json = {
                "paper_id": paper_id, 
                "source": "pmc", 
                "source_id": str(pmc_id), 
                "doi": None, 
                "title": str(title).strip(), 
                "authors": [str(a) for a in authors], 
                "abstract": str(abstract).strip(), 
                "published_date": str(date) if date else None, 
                "updated_date": None, 
                "categories": None, 
                "pdf_url": None, 
                "license": None, 
                "checksum": None, 
                "ingested_at": datetime.now(timezone.utc).isoformat(), 
                "version": 1
                
            }
            
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(record_json, f, indent=2)
                
            ingested += 1
            
            log_event(
                logger=logger, 
                level = logging.INFO, 
                message = "PMC Paper Ingested", 
                paper_id = paper_id
            )
            
        token_element = root.find(".//resumptionToken", ns)
        token = token_element.text.strip() if token_element is not None and token_element.text else None
        
        if not token: 
            break
            
        time.sleep(sleep_records)
        
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "PMC Paginated metadata ingestion complete", 
        ingested = ingested, 
        skipped = skipped
    )
    
if __name__ == "__main__":
    ingest_pmc_metadata_paginated()