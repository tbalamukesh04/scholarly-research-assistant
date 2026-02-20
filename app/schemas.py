from typing import List, Optional, Literal
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: Literal["strict", "exploratory"] = "strict"
    eval_mode: bool = False
    relevant_papers: Optional[List[str]] = None

class Citation(BaseModel):
    citation_id: int
    paper_id: str
    section: str
    text: str
    score: float

class AnswerSentence(BaseModel):
    text: str
    verification_status: Literal["supported", "unsupported"]
    citation_indices: List[int]

class QueryMetrics(BaseModel):
    refused: bool
    refusal_reason: Optional[str] = None
    confidence_score: float = 0.0
    total_latency: float
    retrieval_latency: float
    llm_latency: float
    retrieved_chunks: int = 0
    truncated: bool = False
    dropped_sentences: int = 0

class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    answer_sentences: List[AnswerSentence] = []
    citations: List[Citation] = []
    dataset_hash: str
    index_hash: Optional[str] = None
    run_id: Optional[str] = None
    metrics: QueryMetrics