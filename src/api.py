"""FastAPI wrapper around the RAG pipeline.

Endpoints:
  POST /upload  - upload a PDF, ingest it, return stats
  POST /ask     - ask a question against the loaded PDF
  GET  /health  - liveness check
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

# Single shared pipeline instance
pipe = RAGPipeline()
state = {"loaded": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # cleanup if needed


app = FastAPI(
    title="RAG Doc Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: List[int]
    retrieved: List[dict]


@app.get("/health")
def health():
    return {"status": "ok", "loaded": state["loaded"]}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    # Save to a temp location (data/tmp)
    tmp_dir = Path("data/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename

    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        count = pipe.ingest(str(tmp_path))
        state["loaded"] = True
        return {"filename": file.filename, "chunks": count}
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not state["loaded"]:
        raise HTTPException(400, "No document loaded. Call /upload first.")
    try:
        result = pipe.ask(req.question, top_k=req.top_k)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieved=result["retrieved"],
        )
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")