from typing import List, Optional, Literal
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: Literal["strict", "exploratory"] = "strict"
    eval_mode: bool = False
    relevant_papers: Optional[List[str]] = None
    
class Citation(BaseModel):
    paper_id: str
    chunk_id: str 
    section: str 
    score: float
    
class QueryMetrics(BaseModel):
    total_latency: float
    retrieval_latency: float
    llm_latency: float
    retrieved_chunks: int
    refused: int
    confidence_score: float = 0.0
    truncated: bool = False
    dropped_sentences: int = 0
    
class QueryResponse(BaseModel):
    query: str
    answer: Optional[str]
    citations: List[Citation]
    retrieval_only: bool
    dataset_hash: str
    metrics: QueryMetrics
    