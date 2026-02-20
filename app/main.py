import time
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.dependencies import get_dataset_hash, get_retriever, load_state
from app.schemas import Citation, QueryRequest, QueryResponse, QueryMetrics, AnswerSentence
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
    dataset_hash: str = Depends(get_dataset_hash)
    
):
    start_time = time.time()
    metrics_tracker = RequestMetrics()
    
    try:
        result = answer(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            retriever=retriever,
            eval_mode = req.eval_mode, 
            relevant_papers = req.relevant_papers
        )
        
        total_time = metrics_tracker.total_time()
        
        ans_text = result.get("answer")
        raw_citations = result.get("citations", [])
        raw_sentences = result.get("answer_sentences", [])
        
        # Convert dictionary citations to Pydantic models
        formatted_citations = [Citation(**c) for c in raw_citations]
        
        # Convert dictionary sentences to Pydantic models
        formatted_sentences = [AnswerSentence(**s) for s in raw_sentences]
        
        duration = time.time() - start_time
        
        log_event(
                    server_logger,
                    logging.INFO,
                    "Query Processed",
                    duration=duration,
                    citation_count=len(formatted_citations),
                    has_answer=ans_text is not None,
                )
        
        # Extract metrics
        res_metrics = result.get("metrics", {})
        
        queryMetrics = QueryMetrics(
            refused = res_metrics.get("refusal_triggered", 0.0) > 0.5,
            refusal_reason = res_metrics.get("refusal_reason"),
            confidence_score = res_metrics.get("confidence_score", 0.0),
            total_latency = total_time, 
            retrieval_latency = res_metrics.get("retrieval_latency", 0.0),
            llm_latency = res_metrics.get("llm_latency", 0.0), 
            retrieved_chunks = res_metrics.get("retrieved_chunks", 0),
            truncated = res_metrics.get("truncated", False), 
            dropped_sentences = res_metrics.get("unaligned_sentences", 0)
        )

        return QueryResponse(
            query=req.query,
            answer=ans_text,
            answer_sentences=formatted_sentences,
            citations=formatted_citations,
            dataset_hash=dataset_hash,
            index_hash=result.get("index_hash"),
            run_id=result.get("run_id"),
            metrics = queryMetrics
        )

    except Exception as e:
        import traceback
        traceback.print_exc() 
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