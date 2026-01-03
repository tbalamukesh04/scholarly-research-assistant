from typing import List, Optional, Literal
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: Literal["strict", "exploratory"] = "strict"
    
class Citation(BaseModel):
    paper_id: str
    chunk_id: str 
    section: str 
    score: float
    
class QueryResponse(BaseModel):
    query: str
    answer: Optional[str]
    citations: List[Citation]
    retrieval_only: bool
    dataset_hash: str
    