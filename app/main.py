from fastapi import FastAPI, Depends, HTTPException
from app.schemas import QueryRequest, QueryResponse, Citation
from app.dependencies import load_state, get_dataset_hash, get_retriever

from pipelines.rag.answer import answer as rag_answer

app = FastAPI(
    title = "Scholarly Research Assistant", 
    version = "1.0.0", 
    description = "Citation aware RAG System for scholarly documents"
)

@app.on_event("startup")
def startup_event():
    load_state()
    
@app.post("/query", response_model = QueryResponse)
def query(
    req: QueryRequest, 
    retriever=Depends(get_retriever), 
    dataset_hash: str = Depends(get_dataset_hash),
):
    try:
        result = rag_answer(
            query = req.query,
            top_k = req.top_k,
            mode = req.mode,
            retriever = retriever,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return QueryResponse(
        query=req.query, 
        answer=result.get("answer"),
        citations = [
            Citation(
                paper_id=c.split(":")[0],
                chunk_id = c, 
                section = c.split(":")[1],
                score=0.0,
            )
            for c in result.get("citations", [])
        ],
        retrieval_only = result.get("answer") is None, 
        dataset_hash=dataset_hash
    )