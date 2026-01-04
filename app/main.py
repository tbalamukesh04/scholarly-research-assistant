from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.dependencies import get_dataset_hash, get_retriever, load_state
from app.schemas import Citation, QueryRequest, QueryResponse
from pipelines.rag.answer import answer 


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events for the application.
    """
    load_state()
    yield


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
    try:
        result = answer(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            retriever=retriever,
        )
        
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

        return QueryResponse(
            query=req.query,
            answer=ans_text,
            citations=formatted_citations,
            retrieval_only=ans_text is None,
            dataset_hash=dataset_hash,
        )

    except Exception as e:
        # Print full stack trace to terminal
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
UI_DIR = Path(__file__).parent / "ui"

app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

@app.get("/", response_class=HTMLResponse)
def root():
    return (UI_DIR / "index.html").read_text(encoding="utf-8")