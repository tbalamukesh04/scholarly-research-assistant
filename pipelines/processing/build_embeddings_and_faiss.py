import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logging import setup_logger, log_event
from utils.helper_functions import normalize

CHUNKS_DIR = Path("data/processed/chunks")
OUT_DIR = Path("data/processed/faiss")
INDEX_PATH = OUT_DIR / "index.faiss"
META_PATH = OUT_DIR / "index_meta.json"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_chunks():
    texts = []
    meta = []
    for p in CHUNKS_DIR.glob("*.json"):
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
        
def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name = "Embeddings_FAISS",
        log_dir = "logs",
        level = logging.INFO
    )
    
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "Starting embedding + FAISS Build"
    )
    
    texts, meta = load_chunks()
    if not texts:
        log_event(
            logger = logger,
            level = logging.WARNING, 
            message = "No Text Chunks found!! Aborting Build",
        )
        return 
        
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts, 
        batch_size = 64, 
        show_progress_bar = True, 
        normalize_embeddings = False
    )
    
    emb = normalize(np.asarray(emb).astype("float32"))
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    log_event(
        logger = logger, 
        level = logging.INFO, 
        message = "FAISS Index Built", 
        vectors = index.ntotal, 
        dim = dim
    )
    
if __name__ == "__main__":
    build()
        