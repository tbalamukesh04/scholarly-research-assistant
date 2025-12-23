import json
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
# import yaml

from utils.helper_functions import load_yaml
from utils.logging import setup_logger, log_event

#------------------------------------------------
#----------------Utilities-----------------------
#------------------------------------------------

def sha256_file(path: Path) -> str:
    '''
    Returns specific 64-bit sha string for a file.
    Args:
        path (Path): File path to be calculated.
    
    Returns:
        str: Hexadecimal hash string.'''
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
        return f"sha256: {hasher.hexdigest()}"
        
#----------------------
#----Core Logic--------
#----------------------

def acquire_arxiv_pdfs():
    '''
    Performs the pdf downloading process and stores.
    '''
    project_cfg = load_yaml("configs/project.yaml")
    ingestion_cfg = load_yaml("configs/ingestion.yaml")
    
    log_dir = project_cfg["paths"]["log_root"]
    metadata_dir = Path(ingestion_cfg["storage"]["metadata_pdf_dir"])
    pdf_dir = Path(ingestion_cfg["storage"]["raw_pdf_dir"])
    
    pdf_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="arxiv_pdf_acquisition", 
        log_dir = log_dir, 
        level=logging.INFO
    )
    
    metadata_files = list(metadata_dir.glob("*.json"))
    
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message="Starting PDF Acquisition", 
        total_metadata=len(metadata_files)
    )
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for metapath in metadata_files:
        with metapath.open("r", encoding="utf-8") as f:
            record = json.load(f)
            
        paper_id = record["paper_id"]
        pdf_url = record.get("pdf_url")
        
        if not pdf_url:
            skipped += 1
            log_event(
                logger=logger, 
                level = logging.WARNING,
                message= "No PDF URL!!; skipping", 
                paper_id = paper_id
            )
            continue
        
        pdf_path = pdf_dir / f'{paper_id}.pdf'
        
        if record.get("checksum") and pdf_path.exists():
            skipped += 1
            continue
        
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                raise ValueError(f"Non-pdf Content: {content_type}")
            
            pdf_path.write_bytes(response.content)
            
            checksum = sha256_file(pdf_path)
            
            record["checksum"] = checksum
            record["pdf_acquired_at"] = datetime.now(timezone.utc).isoformat()
            
            tmp_path = metapath.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                
            tmp_path.replace(metapath)
                
            downloaded += 1
            
            log_event(
                logger=logger, 
                level=logging.INFO,
                message="PDF Downloaded and checksummed!!",
                paper_id = paper_id, 
                checksum = checksum
            )
            
            time.sleep(1)
            
        except Exception as e:
            failed += 1
            log_event(
                logger = logger, 
                level = logging.ERROR, 
                message = "PDF Acquisition Failed!!",
                paper_id = paper_id, 
                error = str(e)
            )
            
    log_event(
        logger=logger, 
        level=logging.INFO, 
        message="arxiv PDF Acquisition Complete!!",
        downloaded = downloaded, 
        failed = failed, 
        skipped = skipped
    )
    
    
if __name__ == "__main__":
    acquire_arxiv_pdfs()