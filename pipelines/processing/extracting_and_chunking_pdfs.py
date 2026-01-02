import json
import hashlib
from pathlib import Path
from collections import defaultdict
import pdfplumber
import logging
from utils.logging import setup_logger, log_event

PDF_DIR = Path("data/raw/pdfs")
META_DIR = Path("data/raw/metadata")
OUT_DIR = Path("data/processed/chunks")

SECTION_HEADERS = [
    "abstract", "introduction", "background", "methods", 
    "methodology", "results", "discussion", "conclusion", 
    "references"
]

def normalize(s:str) -> str:
    return s.lower().strip()
    
def detect_section(line:str):
    l = normalize(line)
    for h in SECTION_HEADERS:
        if l.startswith(h):
            return h
    return None
    
def chunk_text(text: str, max_words: int=200):
    words = text.split()
    for i in range(0, len(text), max_words):
        yield " ".join(words[i:i + max_words])
        
def sha256_words(text: str)-> str:
    return "sha256: "+ hashlib.sha256(text.encode("utf-8")).hexdigest()
    
def extract_and_chunk():
    logger = setup_logger(
        name = "pdf_extraction_chunking",
        level = logging.INFO, 
        log_dir = "logs"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "Starting PDF extraction and chunking"
    )
    processed = 0
    skipped = 0
    failed = 0
    
    for meta_path in META_DIR.glob("*.json"):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            
        paper_id = meta["paper_id"]
        pdf_path = PDF_DIR / f"{paper_id}.pdf"
        out_path = OUT_DIR / f"{paper_id}.json"
        
        if not pdf_path.exists():
            log_event(
                logger = logger, 
                level = logging.WARNING,
                message = "PDF Missing, Skipping Paper"
            )
            failed += 1
        if out_path.exists():
            skipped += 1
            continue
            
        sections = defaultdict(list)
        current_section = "unknown"
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    for line in text.splitlines():
                        sec = detect_section(line)
                        if sec: 
                            current_section = sec
                            continue
                        sections[current_section].append(line)
                        
        except Exception as e:
            failed += 1
            log_event(
                logger = logger, 
                level = logging.ERROR, 
                message = "PDF Extraction failed!!",
                paper_id = paper_id, 
                error = str(e)
            )
                    
        structured = {
            "paper_id" : paper_id, 
            "source": meta["source"], 
            "sections" : []
        }
        
        for sec, lines in sections.items():
            joined = " ".join(lines)
            chunks = []
            for idx, chunk in enumerate(chunk_text(joined)):
                chunks.append({
                    "chunk_id": f"{paper_id}::sec::{sec}::chunk::{idx}",
                    "text": chunk,
                    "order": idx, 
                    "token_est": len(chunk.split()) 
                })
                
            structured["sections"].append({
                "section": sec.lower(), 
                "chunks": chunks
            })
        log_event(
            logger = logger, 
            level = logging.INFO, 
            message = "PDF Extracted and Chunked Successfully", 
            paper_id = paper_id, 
            sections = len(structured["sections"])
        )
        
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)
            processed += 1
            
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "PDF Extraction and Chunking Complete", 
        processed = processed, 
        skipped = skipped, 
        failed = failed
    )       
            
if __name__ == "__main__":
    extract_and_chunk()
                    