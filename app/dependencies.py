import json
from pathlib import Path

from pipelines.retrieval.search import Retriever
from pipelines.rag.answer import answer 

DATASET_METADATA = Path("data/versions/dataset_manifest.json")
INDEX_METADATA = Path("data/indexes/index_manifest.json")

class AppState:
    retriever: Retriever
    dataset_hash: str
    
state = AppState()

def load_state():
    with DATASET_METADATA.open("r", encoding="utf-8") as f:
        dataset_metadata = json.load(f)
        
    dataset_hash = dataset_metadata["dataset_hash"]
        
    with INDEX_METADATA.open("r", encoding="utf-8") as f:
        index_metadata = json.load(f)
        
    if index_metadata["dataset_hash"] != dataset_hash:
        raise RuntimeError(
            "Dataset hash mismatch between processed data and FAISS Index"
        )
        
    retriever = Retriever()
    
    state.retriever = retriever
    state.dataset_hash = dataset_hash
    

def get_retriever() -> Retriever:
    return state.retriever

def get_dataset_hash() -> str:
    return state.dataset_hash