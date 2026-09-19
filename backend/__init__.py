"""
Hybrid RAG Backend Package.
Exposes common ingestion, vector storage, graph storage, hybrid retrieval, and guardrails.
"""

import sys
from pathlib import Path

# Auto-link virtual environment site-packages if running outside venv
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_venv_site = _ROOT / ".venv" / "lib"
if _venv_site.exists():
    for _p in _venv_site.glob("python*/site-packages"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from backend.common.pipeline import CommonIngestionPipeline
from backend.common.document_loader import DocumentLoader
from backend.common.text_splitter import CommonTextSplitter

from backend.vector_pipeline.pipeline import VectorStoragePipeline
from backend.vector_pipeline.embeddings import get_embedding_model
from backend.vector_pipeline.faiss_storage import FAISSVectorStoreManager

from backend.graph_pipeline.pipeline import GraphStoragePipeline
from backend.graph_pipeline.entity_extractor import EntityExtractor
from backend.graph_pipeline.neo4j_storage import Neo4jStorageManager

from backend.retrieval.hybrid_pipeline import HybridRetrievalPipeline
from backend.retrieval.vector_retriever import VectorRetriever
from backend.retrieval.graph_retriever import GraphRetriever

from backend.guardrails.manager import HybridRAGGuardrailManager

__all__ = [
    "CommonIngestionPipeline",
    "DocumentLoader",
    "CommonTextSplitter",
    "VectorStoragePipeline",
    "get_embedding_model",
    "FAISSVectorStoreManager",
    "GraphStoragePipeline",
    "EntityExtractor",
    "Neo4jStorageManager",
    "HybridRetrievalPipeline",
    "VectorRetriever",
    "GraphRetriever",
    "HybridRAGGuardrailManager",
]
