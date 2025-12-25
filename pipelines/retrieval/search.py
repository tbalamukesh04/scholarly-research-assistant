import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logging import log_event, setup_logger
from utils.helper_functions import normalize

FAISS_DIR = Path("data/processed/faiss")
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "index_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_DIR = Path("data/processed/chunks")

def attach_text(results):
    cache = {}
    for r in results["results"]:
        pid = r["paper_id"]
        if pid not in cache:
            with (CHUNKS_DIR/ f"{pid}.json").open("r", encoding="utf-8") as f:
                cache[pid] = json.load(f)
               
        for sec in cache[pid]["sections"]:
           if sec["section"] == r["section"]:
               for ch in sec["chunks"]:
                   if ch["chunk_id"] == r["chunk_id"]:
                       r["text"] = r["text"]
                       break
    return results
    
class Retriever:
    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.logger = setup_logger(
            name = "retrieval", 
            log_dir = "./logs",
            level = logging.INFO
        )
        
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        
        with META_PATH.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
            
        log_event(
            logger = self.logger, 
            level = logging.INFO, 
            message = "Retriever Initialized",
            vectors = self.index.ntotal
        )
        
    def search(self, query: str):
        q_emb = self.model.encode([query], normalize_embeddings=False)
        q_emb = normalize(np.asarray(q_emb).astype("float32"))
        
        scores, idxs = self.index.search(q_emb, self.top_k)
        
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            m = self.meta[idx]
            results.append({
                "score": float(score),
                "chunk_id": m["chunk_id"], 
                "paper_id": m["paper_id"], 
                "source": m["source"], 
                "section" : m["section"],
                "order" : m["order"], 
                "text": None
            })
            
        return {
            "query": query, 
            "results": results
        }
        
if __name__ == "__main__":
    r = Retriever(top_k = 5)
    out = r.search("transformer attention mechanism")
    out = attach_text(out)
    
    print(json.dumps(out, indent=2))