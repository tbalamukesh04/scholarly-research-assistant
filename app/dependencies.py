import json
from pathlib import Path
from typing import Optional
import logging

from pipelines.retrieval.search import Retriever
from utils.logging import log_event, setup_logger

DATASET_METADATA = Path("data/versions/dataset_manifest.json")
INDEX_METADATA = Path("data/indexes/index_manifest.json")

# Initialize a specific logger for this module to avoid circular imports
dep_logger = setup_logger(name="dependencies", log_dir="./logs", level=logging.INFO)

class AppState:
    retriever: Optional[Retriever] = None
    dataset_hash: str = "unknown"
    
state = AppState()

def load_state():
    try:
        if not DATASET_METADATA.exists():
             log_event(dep_logger, logging.WARNING, "Dataset manifest not found", path=str(DATASET_METADATA))
             # Fallback for dev
             state.retriever = Retriever()
             state.dataset_hash = "dev_mode"
             return

        with DATASET_METADATA.open("r", encoding="utf-8") as f:
            dataset_metadata = json.load(f)
            
        dataset_hash = dataset_metadata.get("dataset_hash", "unknown_hash")
        
        # Check index manifest if it exists
        if INDEX_METADATA.exists():
            with INDEX_METADATA.open("r", encoding="utf-8") as f:
                index_metadata = json.load(f)
            
            idx_hash = index_metadata.get("dataset_hash")
            if idx_hash and idx_hash != dataset_hash:
                log_event(dep_logger, logging.WARNING, "Hash Mismatch", 
                          dataset_hash=dataset_hash, index_hash=idx_hash)
        else:
             log_event(dep_logger, logging.WARNING, "Index manifest not found", path=str(INDEX_METADATA))

        # Initialize Retriever
        retriever = Retriever()
        
        state.retriever = retriever
        state.dataset_hash = dataset_hash
        
    except Exception as e:
        log_event(dep_logger, logging.ERROR, "State Load Failed", error=str(e))
        # Emergency Fallback
        state.retriever = Retriever()
        state.dataset_hash = "emergency_fallback"

def get_retriever() -> Retriever:
    if state.retriever is None:
        # Lazy load attempt if startup failed
        load_state()
    return state.retriever

def get_dataset_hash() -> str:
    return state.dataset_hash