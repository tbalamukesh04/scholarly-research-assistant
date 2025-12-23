import json
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List
from pathlib import Path

import requests
import yaml
from utils.logging import log_event, setup_logger
from utils.helper_functions import load_yaml, compute_paper_id

def ingest_pmc_metadata():
    '''
    Orchestrate metadata ingestion from PubMed Central
    '''
    project_cfg = load_yaml("configs/project.yaml")
    ingestion_cfg = load_yaml("configs/ingestion.yaml")
    
    log_dir = project_cfg["paths"]["log_root"]
    metadata_dir = Path(ingestion_cfg["storage"]["metadata_dir"])
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name = "pmc_ingestion",
        log_dir= log_dir,
        level=logging.INFO
    )
    
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
    params = {
        "verb": "ListRecords", 
        "metadataPrefix": "oai_dc",
        "set": "pmc-open"
    }
    
    log_event(
        logger=logger,
        level = logging.INFO, 
        message = "Starting PMC Metadata Ingestion", 
    )
    
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    
    from xml.etree import ElementTree as ET 
    root = ET.fromstring(response.text)
    
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/", 
        "dc": "http://purl.org/dc/elements/1.1/"
    }
    
    ingested = 0
    skipped = 0
    
    for record in root.findall(".//oai:record", ns):
        header = record.find("oai:header", ns)
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
            
        meta = record.find("oai:metadata", ns)
        if meta is None:
            continue
            
        dc = meta.find("dc:dc", ns)
        if dc is None:
            continue
            
        title = dc.findtext("dc:title", default="")
        abstract = dc.findtext("dc:description", default="")
        authors = [e.text for e in dc.findall("dc:creator", ns) if e.text]
        date = dc.findtext("dc:date", default=None)
        
        record_json ={
            "paper_id": paper_id, 
            "source": "pmc", 
            "source_id": pmc_id, 
            "doi": None, 
            "title": str(title).strip(), 
            "authors": [str(author) for author in authors],
            "abstract": str(abstract).strip(), 
            "published_date": str(date) if date else None, 
            "updated_date": None, 
            "categories": [], 
            "pdf_url": None, 
            "license": None, 
            "checksum": None, 
            "ingested_at": datetime.now(timezone.utc).isoformat(), 
            "version": 1
        }
        
        with meta_path.open("w", encoding = "utf-8") as f:
            json.dump(record_json, f, indent=2)
            
        ingested += 1
        log_event(
            logger = logger, 
            level = logging.INFO, 
            message="Ingested PMC Metadata", 
            paper_id = paper_id
        )
        time.sleep(0.5)
        
    log_event(
        logger=logger, 
        level = logging.INFO,
        message = "PMC Metadata Ingestion Complete",
        ingested = ingested, 
        skipped = skipped
    )
    
if __name__ == "__main__":
    ingest_pmc_metadata()