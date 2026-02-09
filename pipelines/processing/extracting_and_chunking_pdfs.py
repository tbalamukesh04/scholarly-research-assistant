import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
import pdfplumber
import logging
from utils.logging import setup_logger, log_event
from utils.helper_functions import load_yaml # Added to read params

# Global constants (can be moved to params if needed, but keeping simple for now)
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
    
def chunk_text(text: str, max_words: int):
    words = text.split()
    for i in range(0, len(text), max_words):
        yield " ".join(words[i:i + max_words])
        
def sha256_words(text: str)-> str:
    return "sha256: "+ hashlib.sha256(text.encode("utf-8")).hexdigest()
    
def extract_and_chunk(pdf_dir: Path, meta_dir: Path, out_dir: Path):
    # Load params to get chunk size
    params = load_yaml("params.yaml")
    chunk_size = params["processing"]["chunk_size"]
    
    logger = setup_logger(
        name = "pdf_extraction_chunking",
        level = logging.INFO, 
        log_dir = "logs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "Starting PDF extraction and chunking",
        chunk_size = chunk_size
    )
    processed = 0
    skipped = 0
    failed = 0
    
    # We iterate metadata files to drive processing
    # If metadata is missing, we fall back to PDF files directly
    if meta_dir.exists():
        files = list(meta_dir.glob("*.json"))
    else:
        files = list(pdf_dir.glob("*.pdf"))

    for item_path in files:
        if meta_dir.exists():
            with item_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            paper_id = meta["paper_id"]
            pdf_path = pdf_dir / f"{paper_id}.pdf"
            source = meta.get("source", "unknown")
        else:
            # Fallback if no metadata (Direct PDF processing)
            if item_path.suffix != ".pdf": continue
            paper_id = item_path.stem
            pdf_path = item_path
            source = "unknown"

        out_path = out_dir / f"{paper_id}.json"
        
        if not pdf_path.exists():
            # log_event(logger=logger, level=logging.WARNING, message="PDF Missing", paper_id=paper_id)
            failed += 1
            continue

        # Force regenerate if chunk size changes? 
        # DVC handles the stage invalidation, so we just overwrite if run.
        # But we can check if we want to skip existing identical files.
        # For now, let's write every time to ensure param update takes effect.
        
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
            log_event(logger=logger, level=logging.ERROR, message="Extraction Failed", paper_id=paper_id, error=str(e))
            continue
                    
        structured = {
            "paper_id" : paper_id, 
            "source": source, 
            "sections" : []
        }
        
        for sec, lines in sections.items():
            joined = " ".join(lines)
            chunks = []
            # Use the dynamic chunk_size here
            for idx, chunk in enumerate(chunk_text(joined, max_words=chunk_size)):
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
        
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)
            processed += 1
            
    log_event(logger=logger, level=logging.INFO, message="Complete", processed=processed, skipped=skipped, failed=failed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to raw PDFs")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to save chunks")
    parser.add_argument("--meta_dir", type=Path, default=Path("data/raw/metadata"), help="Path to metadata")
    
    args = parser.parse_args()
    
    extract_and_chunk(args.input_dir, args.meta_dir, args.output_dir)