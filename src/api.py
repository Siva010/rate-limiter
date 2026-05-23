# src/api.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from config import DATA_DIR, TOP_K, HYBRID_ALPHA
from data import load_documents, prepare_chunks
from retrieval import BM25Retriever, PineconeDenseRetriever, HybridRetriever
from pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline on startup."""
    global pipeline
    print("Initializing system...")
    raw_docs = load_documents(DATA_DIR)
    documents = prepare_chunks(raw_docs)
    bm25 = BM25Retriever(documents)
    dense = PineconeDenseRetriever()
    hybrid = HybridRetriever(bm25, dense, HYBRID_ALPHA)
    pipeline = RAGPipeline(hybrid)
    print("System ready")
    yield
    # Cleanup (if needed)
    pipeline = None


app = FastAPI(title="Semantic Search API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "ready": pipeline is not None}


@app.post("/query")
def query_rag(req: QueryRequest):
    if pipeline is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="System not ready")
    answer = pipeline.run(req.query, req.top_k)
    return {
        "query": req.query,
        "answer": answer
    }
