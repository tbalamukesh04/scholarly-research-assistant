import time
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.dependencies import get_dataset_hash, get_retriever, load_state
from app.schemas import Citation, QueryRequest, QueryResponse, QueryMetrics
from pipelines.rag.answer import answer 
from app.metrics import RequestMetrics
from utils.logging import log_event, logging, setup_logger

server_logger = setup_logger(
    name = "server", 
    log_dir = "./logs",
    level = logging.INFO
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events for the application.
    """
    log_event(server_logger, logging.INFO, "Server Setup Initiated")
    try:
        load_state()
        log_event(server_logger, logging.INFO, "Server State Loaded Successfully")
    except Exception as e:
        log_event(server_logger, logging.CRITICAL, "Startup Error", error = str(e))
    yield
    log_event(server_logger, logging.INFO, "Server Shutdown Initiated")

app = FastAPI(
    title="Scholarly Research Assistant",
    version="1.0.0",
    description="Citation aware RAG System for scholarly documents",
    lifespan=lifespan,
)

@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    retriever=Depends(get_retriever),
    dataset_hash: str = Depends(get_dataset_hash),
):
    start_time = time.time()
    metrics_tracker = RequestMetrics()
    
    try:
        result = answer(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            retriever=retriever,
        )
        
        total_time = metrics_tracker.total_time()
        # Determine if answer exists
        ans_text = result.get("answer")
        raw_citations = result.get("citations", [])

        formatted_citations = []
        for c in raw_citations:
            parts = str(c).split(":")
            formatted_citations.append(
                Citation(
                    paper_id=parts[0] if len(parts) > 0 else "unknown",
                    chunk_id=str(c),
                    section=parts[1] if len(parts) > 1 else "unknown",
                    score=0.0,
                )
            )
        
        duration = time.time() - start_time
        
        log_event(
                    server_logger,
                    logging.INFO,
                    "Query Processed",
                    duration=duration,
                    citation_count=len(formatted_citations),
                    has_answer=ans_text is not None,
                )
                
        queryMetrics = QueryMetrics(
            total_latency = total_time, 
            retrieval_latency = result.get("metrics", {}).get("retrieval_latency", 0.0),
            llm_latency = result.get("metrics", {}).get("llm_latency", 0.0), 
            retrieved_chunks = result.get("metrics", {}).get("retrieved_chunks", 0),
            refused = 1 if result.get("metrics", {}).get("refused", False) else 0
        )

        return QueryResponse(
            query=req.query,
            answer=ans_text,
            citations=formatted_citations,
            retrieval_only=ans_text is None,
            dataset_hash=dataset_hash,
            metrics = queryMetrics
        )

    except Exception as e:
        # Print full stack trace to terminal
        import traceback
        traceback.print_exc()  # Keep terminal output for immediate dev visibility
        log_event(
            server_logger,
            logging.ERROR,
            "Query Execution Failed",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))
        
UI_DIR = Path(__file__).parent / "ui"

app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

@app.get("/", response_class=HTMLResponse)
def root():
    return (UI_DIR / "index.html").read_text(encoding="utf-8")