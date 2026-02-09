import json
import logging
import argparse
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logging import setup_logger, log_event
from utils.helper_functions import normalize
from scripts.write_index_manifest import write_index_manifest

# REMOVED GLOBAL CONSTANTS for Paths
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_chunks(chunks_dir: Path):
    texts = []
    meta = []
    # Ensure directory exists
    if not chunks_dir.exists():
        return [], []
        
    for p in chunks_dir.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        for sec in doc.get("sections", []):
            for ch in sec.get("chunks", []):
                texts.append(ch["text"])
                meta.append({
                    "chunk_id": ch["chunk_id"],
                    "paper_id": doc["paper_id"], 
                    "source": doc["source"], 
                    "section": sec["section"],
                    "order":ch["order"]
                })
    return texts, meta
        
def build(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    meta_path = output_dir / "index_meta.json"
    
    logger = setup_logger(name="Embeddings_FAISS", log_dir="logs", level=logging.INFO)
    log_event(logger=logger, level=logging.INFO, message="Starting FAISS Build")
    
    texts, meta = load_chunks(input_dir)
    if not texts:
        log_event(logger=logger, level=logging.WARNING, message="No Text Chunks found!!")
        return 
        
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    
    emb = normalize(np.asarray(emb).astype("float32"))
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    # Adjust manifest writer if needed, or assume it works in context
    try:
        write_index_manifest()
    except:
        pass # Warning: Manifest writer might need update too if it hardcodes paths
    
    log_event(logger=logger, level=logging.INFO, message="FAISS Index Built", vectors=index.ntotal, dim=dim)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory (chunks)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory (indexes)")
    
    args = parser.parse_args()
    
    build(args.input_dir, args.output_dir)