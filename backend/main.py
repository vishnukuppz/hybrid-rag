import os
import sys
import time
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root and virtual environment site-packages are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

venv_site_packages = PROJECT_ROOT / ".venv" / "lib"
if venv_site_packages.exists():
    for p in venv_site_packages.glob("python*/site-packages"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from backend import (
    CommonIngestionPipeline,
    VectorStoragePipeline,
    GraphStoragePipeline,
    HybridRetrievalPipeline,
    HybridRAGGuardrailManager,
)

load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_rag_api")

app = FastAPI(
    title="Hybrid RAG API",
    description="FastAPI backend for Hybrid RAG: Common Ingestion, FAISS Vector DB, Neo4j Knowledge Graph, and Guardrails",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline singletons
_retrieval_pipeline: Optional[HybridRetrievalPipeline] = None
_guardrail_manager: Optional[HybridRAGGuardrailManager] = None


def get_retrieval_pipeline() -> HybridRetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = HybridRetrievalPipeline(
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


def check_faiss_status() -> Dict[str, Any]:
    faiss_dir = PROJECT_ROOT / "faiss_index"
    faiss_file = faiss_dir / "index.faiss"
    pkl_file = faiss_dir / "index.pkl"
    available = faiss_file.exists() and pkl_file.exists()
    size_bytes = 0
    if available:
        try:
            size_bytes = faiss_file.stat().st_size + pkl_file.stat().st_size
        except Exception:
            pass
    return {
        "available": available,
        "path": str(faiss_dir),
        "size_bytes": size_bytes,
    }


# =============================================================================
# Request & Response Models
# =============================================================================
class ChatRequest(BaseModel):
    question: str = Field(..., description="User query for the Hybrid RAG system")
    enable_guardrails: bool = Field(default=True, description="Enable safety moderation and jailbreak checks")
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
    """Returns availability of the FAISS Vector Database and Neo4j Knowledge Graph."""
    faiss_info = check_faiss_status()
    retriever = get_retrieval_pipeline()
    graph_available = retriever.graph_retriever.is_available()

    return {
        "faiss_db": faiss_info,
        "neo4j_graph": {"available": graph_available},
        "ready_for_chat": faiss_info["available"],
    }


@app.post("/api/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
):
    """
    Ingests a document through Common Preprocessing -> FAISS Vector DB & Neo4j Knowledge Graph.
    Accepts either an uploaded file or `use_sample=True` to load spotify_web_app_architecture.pdf.
    """
    global _retrieval_pipeline
    t_start = time.time()
    logs: List[str] = []
    temp_file_path: Optional[Path] = None

    try:
        if use_sample:
            sample_path = PROJECT_ROOT / "docs" / "spotify_web_app_architecture.pdf"
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail="Default sample PDF not found at docs/spotify_web_app_architecture.pdf")
            target_path = sample_path
            filename = sample_path.name
        elif file is not None:
            filename = file.filename or "uploaded_document"
            temp_dir = Path(tempfile.gettempdir()) / "hybrid_rag_api_uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file_path = temp_dir / filename
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            target_path = temp_file_path
        else:
            raise HTTPException(status_code=400, detail="Must provide an uploaded file or set use_sample=true")

        # 1. Common Ingestion
        logger.info(f"Ingesting: {target_path}")
        common = CommonIngestionPipeline(chunk_size=600, chunk_overlap=100, mode="auto", verbose=True)
        docs, chunks = common.run(target_path)

        if not docs or not chunks:
            raise HTTPException(status_code=422, detail="No content could be extracted from the document")

        logs.append(f"📄 Loaded **{len(docs)}** document sections from `{filename}`")
        logs.append(f"✂️ Split into **{len(chunks)}** contextual chunks (chunk_size=600, overlap=100)")

        # 2. Vector Pipeline (FAISS)
        faiss_save_dir = PROJECT_ROOT / "faiss_index"
        vec_pipe = VectorStoragePipeline(
            embedding_provider="openai",
            save_directory=faiss_save_dir,
            verbose=True,
        )
        vec_res = vec_pipe.run_on_chunks(chunks)
        total_vectors = vec_res.get("total_vectors_in_store", len(chunks))
        logs.append(f"🧠 Built FAISS Vector DB: **{total_vectors}** vectors indexed at `faiss_index/`")

        # 3. Graph Pipeline (Neo4j)
        graph_pipe = GraphStoragePipeline(max_concurrency=5, verbose=True)
        graph_res = graph_pipe.run_on_chunks(chunks)
        nodes = graph_res.get("neo4j_total_nodes", 0)
        rels = graph_res.get("neo4j_total_relationships", 0)
        logs.append(f"🕸️ Built Neo4j Knowledge Graph: **{nodes}** nodes, **{rels}** relationships")

        # Invalidate cached retrieval pipeline so new FAISS index is reloaded
        _retrieval_pipeline = None

        t_total = time.time() - t_start
        logs.append(f"🎉 Ingestion completed in **{t_total:.2f} seconds**.")

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
        # Clean up temporary uploaded file if created
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Executes a Hybrid RAG query across Vector DB & Neo4j Knowledge Graph,
    with safety guardrails evaluation.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Guardrails evaluation
    if request.enable_guardrails:
        guardrails = get_guardrail_manager()
        eval_result = guardrails.evaluate(question)

        if not eval_result.get("is_safe", True):
            category = eval_result.get("category", "off_topic")
            reason = eval_result.get("refusal_reason", "Out of domain")
            refusal_text = eval_result.get(
                "suggested_response",
                "I am specialized in technical system architecture and Spotify web app specifications. Please ask questions related to services, APIs, databases, or routes.",
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

    # Retrieval and synthesis
    retriever = get_retrieval_pipeline()
    result = retriever.run(
        question=question,
    )

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


def main():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🚀 Starting Hybrid RAG FastAPI server on http://{host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
