"""
Hybrid RAG Backend Package.
All imports use non-prefixed paths since the Docker build context is backend/
and PYTHONPATH=/app maps directly to the backend/ folder contents.
"""

from common.pipeline import CommonIngestionPipeline
from common.document_loader import DocumentLoader
from common.text_splitter import CommonTextSplitter

from vector_pipeline.pipeline import VectorStoragePipeline
from vector_pipeline.embeddings import get_embedding_model
from vector_pipeline.faiss_storage import FAISSVectorStoreManager

from graph_pipeline.pipeline import GraphStoragePipeline
from graph_pipeline.entity_extractor import EntityExtractor
from graph_pipeline.neo4j_storage import Neo4jStorageManager

from retrieval.hybrid_pipeline import HybridRetrievalPipeline
from retrieval.vector_retriever import VectorRetriever
from retrieval.graph_retriever import GraphRetriever

from guardrails.manager import HybridRAGGuardrailManager

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
