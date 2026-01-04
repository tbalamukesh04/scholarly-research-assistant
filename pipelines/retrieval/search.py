import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logging import log_event, setup_logger
from utils.helper_functions import normalize
from pipelines.retrieval.hydrate import attach_text

FAISS_DIR = Path("data/processed/faiss")
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "index_meta.json"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

CHUNKS_DIR = Path("data/processed/chunks")
    
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
        
    def search(self, query: str)-> dict:
        '''
        Searches for relevant documents based on a query.
        Args:
            query (str): The query to search for.
        Returns:
            dict: A dictionary containing the query and the results.
        '''
        q_emb = self.model.encode([query], normalize_embeddings=False)
        q_emb = normalize(np.asarray(q_emb).astype("float32"))
        
        scores, idxs = self.index.search(q_emb, self.top_k)
        
        results = []
        MIN_SCORE = 0.0
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
        
        results = [
            r for r in results
            if r["score"] > MIN_SCORE
        ]
            
        return {
            "results": results
        }
        
if __name__ == "__main__":
    r = Retriever(top_k = 5)
    out = r.search("""We begin by developing DL models tailored for specific XR ap
    plications, focusing on classifying cybersickness, user emotions,
    and activity. """)
    # print(out)
    out = attach_text(out)
    
    print(json.dumps(out, indent=2))