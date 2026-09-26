import os
import sys
import time
import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: works both locally and inside Docker.
#   Locally  → adds backend/ to sys.path so `from common.X` resolves.
#   In Docker → PYTHONPATH=/app already covers /app which IS backend/.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
import logging
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import uvicorn

from common.pipeline import CommonIngestionPipeline
from graph_pipeline.pipeline import GraphStoragePipeline
from retrieval.hybrid_pipeline import HybridRetrievalPipeline
from guardrails.manager import HybridRAGGuardrailManager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_rag_api")

# Internal service URL for the FAISS database container
DATABASE_URL = os.getenv("DATABASE_URL", "http://localhost:8002").rstrip("/")
DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(DEFAULT_DOCS_DIR)))

app = FastAPI(
    title="Hybrid RAG API",
    description="FastAPI backend — ingestion, Neo4j graph, FAISS (via DB service), and guardrails",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_retrieval_pipeline: Optional[HybridRetrievalPipeline] = None
_guardrail_manager: Optional[HybridRAGGuardrailManager] = None


def get_retrieval_pipeline() -> HybridRetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = HybridRetrievalPipeline(
            database_url=DATABASE_URL,
            verbose=True,
        )
    return _retrieval_pipeline


def get_guardrail_manager() -> HybridRAGGuardrailManager:
    global _guardrail_manager
    if _guardrail_manager is None:
        _guardrail_manager = HybridRAGGuardrailManager(
            use_nemo=True,
            verbose=True,
        )
    return _guardrail_manager


def get_faiss_status_from_db() -> Dict[str, Any]:
    """Fetches FAISS index status from the database service."""
    try:
        resp = requests.get(f"{DATABASE_URL}/api/db/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Cannot reach database service at {DATABASE_URL}: {e}")
    return {"available": False, "path": "", "size_bytes": 0, "vector_count": 0}


# =============================================================================
# Request & Response Models
# =============================================================================

class ChatRequest(BaseModel):
    question: str = Field(..., description="User query for the Hybrid RAG system")
    enable_guardrails: bool = Field(default=True, description="Enable safety guardrails")
    top_k_vector: int = Field(default=4, description="Number of vector DB passages to retrieve")


class ChatResponse(BaseModel):
    question: str
    answer: str
    is_refusal: bool = False
    guardrail_status: str = "approved"
    refusal_reason: Optional[str] = None
    vector_evidence: List[Dict[str, Any]] = []
    graph_evidence: Dict[str, Any] = {}
    timing: Dict[str, float] = {}


class IngestionResponse(BaseModel):
    status: str
    file_name: str
    documents_count: int
    chunks_count: int
    vectors_indexed: int
    graph_nodes: int
    graph_relationships: int
    execution_time_seconds: float
    logs: List[str]


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
def root():
    return {
        "app": "Hybrid RAG API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": ["/health", "/api/status", "/api/ingest", "/api/chat"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/status")
def status():
    """Returns availability of the FAISS DB (via database service) and Neo4j graph."""
    faiss_info = get_faiss_status_from_db()
    retriever = get_retrieval_pipeline()
    graph_available = retriever.graph_retriever.is_available()

    return {
        "faiss_db": faiss_info,
        "neo4j_graph": {"available": graph_available},
        "ready_for_chat": faiss_info.get("available", False),
        "database_service_url": DATABASE_URL,
    }


@app.post("/api/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
):
    """
    Ingests a document through:
      1. Common Preprocessing (load + split into chunks)
      2. FAISS Vector DB  → sends chunks to the database service
      3. Neo4j Knowledge Graph  → entity/relationship extraction
    """
    global _retrieval_pipeline
    t_start = time.time()
    logs: List[str] = []
    temp_file_path: Optional[Path] = None

    try:
        # Resolve source file
        if use_sample:
            sample_path = DOCS_DIR / "spotify_web_app_architecture.pdf"
            if not sample_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail="Sample PDF not found at docs/spotify_web_app_architecture.pdf",
                )
            target_path = sample_path
            filename = sample_path.name
        elif file is not None:
            filename = file.filename or "uploaded_document"
            temp_dir = Path(tempfile.gettempdir()) / "hybrid_rag_uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file_path = temp_dir / filename
            with open(temp_file_path, "wb") as buf:
                shutil.copyfileobj(file.file, buf)
            target_path = temp_file_path
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide an uploaded file or set use_sample=true",
            )

        # 1. Common Ingestion (load + split)
        logger.info(f"Starting ingestion: {target_path}")
        common = CommonIngestionPipeline(chunk_size=600, chunk_overlap=100, mode="auto", verbose=True)
        docs, chunks = common.run(target_path)

        if not docs or not chunks:
            raise HTTPException(status_code=422, detail="No content could be extracted from the document")

        logs.append(f"📄 Loaded **{len(docs)}** document sections from `{filename}`")
        logs.append(f"✂️ Split into **{len(chunks)}** contextual chunks (size=600, overlap=100)")

        # 2. Send chunks to FAISS Database Service
        logger.info(f"Sending {len(chunks)} chunks to database service at {DATABASE_URL}")
        chunks_payload = [
            {"page_content": chunk.page_content, "metadata": dict(chunk.metadata)}
            for chunk in chunks
        ]
        try:
            db_resp = requests.post(
                f"{DATABASE_URL}/api/db/store",
                json={"chunks": chunks_payload, "append": False},
                timeout=300,
            )
            db_resp.raise_for_status()
            db_data = db_resp.json()
            total_vectors = db_data.get("total_vectors", len(chunks))
            logs.append(f"🧠 Indexed **{total_vectors}** vectors in FAISS database service")
        except Exception as e:
            logger.error(f"Database service error: {e}")
            raise HTTPException(status_code=502, detail=f"FAISS database service failed: {str(e)}")

        # 3. Graph Pipeline (Neo4j)
        graph_pipe = GraphStoragePipeline(max_concurrency=5, verbose=True)
        graph_res = graph_pipe.run_on_chunks(chunks)
        nodes = graph_res.get("neo4j_total_nodes", 0)
        rels = graph_res.get("neo4j_total_relationships", 0)
        logs.append(f"🕸️ Built Neo4j Knowledge Graph: **{nodes}** nodes, **{rels}** relationships")

        # Invalidate cached retrieval pipeline
        _retrieval_pipeline = None

        t_total = time.time() - t_start
        logs.append(f"🎉 Ingestion completed in **{t_total:.2f} seconds**")

        return IngestionResponse(
            status="success",
            file_name=filename,
            documents_count=len(docs),
            chunks_count=len(chunks),
            vectors_indexed=total_vectors,
            graph_nodes=nodes,
            graph_relationships=rels,
            execution_time_seconds=round(t_total, 2),
            logs=logs,
        )
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Hybrid RAG chat: guardrails → vector retrieval (via DB service) + graph retrieval → LLM synthesis.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Guardrails check
    if request.enable_guardrails:
        guardrails = get_guardrail_manager()
        eval_result = guardrails.evaluate(question)

        if not eval_result.get("is_safe", True):
            category = eval_result.get("category", "off_topic")
            reason = eval_result.get("refusal_reason", "Out of domain")
            refusal_text = eval_result.get(
                "suggested_response",
                "I specialise in technical system architecture. Please ask questions related to services, APIs, databases, or routes.",
            )
            return ChatResponse(
                question=question,
                answer=refusal_text,
                is_refusal=True,
                guardrail_status=category,
                refusal_reason=reason,
                vector_evidence=[],
                graph_evidence={},
                timing={"total_seconds": 0.05},
            )

    # Retrieval + synthesis
    retriever = get_retrieval_pipeline()
    result = retriever.run(question=question)

    v_res = result.get("vector_result", {})
    g_res = result.get("graph_result", {})
    timings = result.get("timings", {})

    return ChatResponse(
        question=question,
        answer=result.get("answer", ""),
        is_refusal=False,
        guardrail_status="approved",
        vector_evidence=v_res.get("chunks", []),
        graph_evidence=g_res,
        timing=timings,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🚀 Starting Hybrid RAG FastAPI server on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
