import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import uvicorn

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faiss_db_service")

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="FAISS Database Service",
    description="Microservice for FAISS vector database — store, search, and status endpoints",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cached vector store
_vector_store: Optional[FAISS] = None


def get_embeddings() -> OpenAIEmbeddings:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def get_faiss_status() -> Dict[str, Any]:
    index_file = FAISS_INDEX_DIR / "index.faiss"
    pkl_file = FAISS_INDEX_DIR / "index.pkl"
    available = index_file.exists() and pkl_file.exists()
    size_bytes = 0
    vector_count = 0
    if available:
        try:
            size_bytes = index_file.stat().st_size + pkl_file.stat().st_size
            if _vector_store is not None:
                vector_count = _vector_store.index.ntotal
        except Exception:
            pass
    return {
        "available": available,
        "path": str(FAISS_INDEX_DIR),
        "size_bytes": size_bytes,
        "vector_count": vector_count,
    }


# =============================================================================
# Request / Response Models
# =============================================================================

class ChunkItem(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}


class StoreRequest(BaseModel):
    chunks: List[ChunkItem]
    append: bool = Field(default=False, description="Append to existing index; false = rebuild")


class StoreResponse(BaseModel):
    status: str
    chunks_stored: int
    total_vectors: int
    elapsed_seconds: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=50)


class SearchResult(BaseModel):
    rank: int
    score: float
    content: str
    section: str = ""
    component: str = ""
    pages: str = ""
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int
    elapsed_seconds: float


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def root():
    return {
        "service": "FAISS Database Service",
        "version": "1.0.0",
        "endpoints": ["/health", "/api/db/status", "/api/db/store", "/api/db/search", "/api/db/clear"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/db/status")
def db_status():
    """Returns current FAISS index availability and stats."""
    return get_faiss_status()


@app.post("/api/db/store", response_model=StoreResponse)
def store_chunks(request: StoreRequest):
    """
    Receive document chunks and build/update the FAISS index.
    Set append=false to rebuild from scratch (default).
    """
    global _vector_store

    if not request.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")

    t_start = time.time()
    logger.info(f"Storing {len(request.chunks)} chunks (append={request.append})")

    try:
        embeddings = get_embeddings()
        documents = [
            Document(page_content=c.page_content, metadata=c.metadata)
            for c in request.chunks
        ]

        index_file = FAISS_INDEX_DIR / "index.faiss"
        if request.append and index_file.exists():
            existing = FAISS.load_local(
                str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )
            existing.add_documents(documents)
            _vector_store = existing
        else:
            _vector_store = FAISS.from_documents(documents, embeddings)

        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _vector_store.save_local(str(FAISS_INDEX_DIR))

        total_vectors = _vector_store.index.ntotal if hasattr(_vector_store, "index") else len(documents)
        elapsed = time.time() - t_start
        logger.info(f"FAISS index saved: {total_vectors} vectors in {elapsed:.2f}s")

        return StoreResponse(
            status="success",
            chunks_stored=len(documents),
            total_vectors=total_vectors,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"Failed to store chunks: {e}")
        raise HTTPException(status_code=500, detail=f"FAISS store failed: {str(e)}")


@app.post("/api/db/search", response_model=SearchResponse)
def search_chunks(request: SearchRequest):
    """
    Perform similarity search on the FAISS index.
    Returns ranked results with L2 distance scores and metadata.
    """
    global _vector_store

    t_start = time.time()
    index_file = FAISS_INDEX_DIR / "index.faiss"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="FAISS index not found. Please ingest documents first.",
        )

    try:
        if _vector_store is None:
            embeddings = get_embeddings()
            _vector_store = FAISS.load_local(
                str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
            )

        docs_and_scores = _vector_store.similarity_search_with_score(request.query, k=request.top_k)

        results = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            meta = doc.metadata
            p_start = meta.get("page_start", meta.get("page", "?"))
            p_end = meta.get("page_end", p_start)
            results.append(SearchResult(
                rank=rank,
                score=round(float(score), 4),
                content=doc.page_content,
                section=meta.get("section", "General"),
                component=meta.get("component", ""),
                pages=f"P{p_start}-P{p_end}",
                metadata=meta,
            ))

        elapsed = time.time() - t_start
        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
            elapsed_seconds=round(elapsed, 3),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.delete("/api/db/clear")
def clear_index():
    """Clears the FAISS index from disk and resets in-memory cache."""
    global _vector_store
    _vector_store = None
    cleared = []
    for fname in ["index.faiss", "index.pkl"]:
        f = FAISS_INDEX_DIR / fname
        if f.exists():
            f.unlink()
            cleared.append(fname)
    return {"status": "cleared", "files_removed": cleared}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
