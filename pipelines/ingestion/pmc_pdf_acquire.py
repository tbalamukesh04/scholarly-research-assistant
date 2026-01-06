import json
import hashlib
import logging
import time
import re
from pathlib import Path
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.helper_functions import load_yaml
from utils.logging import setup_logger, log_event

#------------------------------------------------
#----------------Utilities-----------------------
#------------------------------------------------

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
        return f"sha256: {hasher.hexdigest()}"

def get_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,application/pdf;q=0.8,*/*;q=0.8",
    })
    return session

def resolve_pdf_url(session, initial_url):
    """
    Fetches the initial URL. If it's a PDF, returns content.
    If HTML, scrapes the 'citation_pdf_url' meta tag and fetches that.
    """
    resp = session.get(initial_url, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    
    content_type = resp.headers.get("Content-Type", "").lower()
    
    # Case 1: Direct PDF
    if "pdf" in content_type:
        return resp.content, initial_url
        
    # Case 2: HTML Wrapper -> Extract Real Link
    if "html" in content_type:
        # Use regex to find <meta name="citation_pdf_url" content="...">
        # This is standard HighWire Press / Google Scholar metadata
        match = re.search(r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\'](.*?)["\']', resp.text, re.IGNORECASE)
        if match:
            real_pdf_url = match.group(1)
            # Handle relative URLs if necessary (rare on PMC but possible)
            if not real_pdf_url.startswith("http"):
                # Basic join (assumes root relative)
                base = "https://pmc.ncbi.nlm.nih.gov" 
                real_pdf_url = base + real_pdf_url if real_pdf_url.startswith("/") else base + "/" + real_pdf_url
                
            # Fetch the real PDF
            pdf_resp = session.get(real_pdf_url, timeout=30)
            pdf_resp.raise_for_status()
            
            if "pdf" not in pdf_resp.headers.get("Content-Type", "").lower():
                raise ValueError(f"Extracted URL {real_pdf_url} did not return PDF.")
                
            return pdf_resp.content, real_pdf_url
            
    raise ValueError("Could not resolve actual PDF binary from URL.")

#----------------------
#----Core Logic--------
#----------------------

def acquire_pmc_pdfs():
    project_cfg = load_yaml("configs/project.yaml")
    ingestion_cfg = load_yaml("configs/ingestion.yaml")
    
    log_dir = project_cfg["paths"]["log_root"]
    metadata_dir = Path(ingestion_cfg["storage"]["metadata_dir"])
    pdf_dir = Path(ingestion_cfg["storage"]["raw_pdf_dir"])
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(name="pmc_pdf_acquisition", log_dir=log_dir, level=logging.INFO)
    
    # Identify PMC records
    metadata_files = []
    for f in metadata_dir.glob("*.json"):
        try:
            with f.open("r", encoding="utf-8") as meta_f:
                data = json.load(meta_f)
                if data.get("source") == "pmc":
                    metadata_files.append((f, data))
        except Exception:
            continue
    
    log_event(logger=logger, level=logging.INFO, message="Starting PMC PDF Acquisition", total_metadata=len(metadata_files))
    
    downloaded = 0
    skipped = 0
    failed = 0
    sleep_time = ingestion_cfg["sources"]["pmc"].get("sleep_records", 2.0)
    session = get_session()

    for metapath, record in metadata_files:
        paper_id = record["paper_id"]
        pdf_url = record.get("pdf_url")
        
        if not pdf_url:
            skipped += 1
            continue
        
        pdf_path = pdf_dir / f'{paper_id}.pdf'
        
        if record.get("checksum") and pdf_path.exists():
            skipped += 1
            continue
        
        try:
            # Attempt to resolve and download
            pdf_content, final_url = resolve_pdf_url(session, pdf_url)
            
            pdf_path.write_bytes(pdf_content)
            checksum = sha256_file(pdf_path)
            
            # Update record
            record["checksum"] = checksum
            record["pdf_acquired_at"] = datetime.now(timezone.utc).isoformat()
            # Optionally update the URL to the direct link for future reference
            if final_url != pdf_url:
                record["pdf_url_direct"] = final_url
            
            # Atomic write
            tmp_path = metapath.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tmp_path.replace(metapath)
                
            downloaded += 1
            log_event(logger=logger, level=logging.INFO, message="PDF Downloaded", paper_id=paper_id, checksum=checksum)
            
            time.sleep(sleep_time)
            
        except Exception as e:
            failed += 1
            log_event(logger=logger, level=logging.ERROR, message="PDF Acquisition Failed", paper_id=paper_id, error=str(e))
            
    log_event(logger=logger, level=logging.INFO, message="PMC PDF Acquisition Complete", downloaded=downloaded, failed=failed, skipped=skipped)

if __name__ == "__main__":
    acquire_pmc_pdfs()