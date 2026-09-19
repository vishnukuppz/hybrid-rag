import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.vector_pipeline.embeddings import get_embedding_model

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class VectorRetriever:
    """
    Retrieves relevant semantic document chunks from the FAISS Vector Database.
    Extracts text passages with similarity scores and architectural metadata.
    """

    def __init__(
        self,
        save_directory: Union[str, Path] = "faiss_index",
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        top_k: int = 4,
        verbose: bool = True,
    ):
        cand_path = Path(save_directory)
        if not cand_path.is_absolute() and not (cand_path / "index.faiss").exists() and (PROJECT_ROOT / cand_path / "index.faiss").exists():
            self.save_directory = (PROJECT_ROOT / cand_path).resolve()
        else:
            self.save_directory = cand_path.resolve()
        self.top_k = top_k
        self.verbose = verbose
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        self._vector_store: Optional[FAISS] = None
        self.embeddings = get_embedding_model(
            provider=self.embedding_provider,
            model_name=self.embedding_model,
        )

    def is_available(self) -> bool:
        """Check if FAISS index files exist on disk."""
        index_file = self.save_directory / "index.faiss"
        pkl_file = self.save_directory / "index.pkl"
        return index_file.exists() and pkl_file.exists()

    def get_vector_store(self) -> FAISS:
        """Lazily load and cache the FAISS vector store."""
        if self._vector_store is None:
            if not self.is_available():
                raise FileNotFoundError(f"FAISS index not found at: {self.save_directory}")
            if self.verbose:
                print(f"📂 [VECTOR RETRIEVER] Loading FAISS index from: {self.save_directory}")
            self._vector_store = FAISS.load_local(
                str(self.save_directory),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._vector_store

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute similarity search on the vector store.
        Returns retrieved chunks, metadata, scores, and formatted context text.
        """
        k = top_k or self.top_k
        start_time = time.time()

        if not self.is_available():
            return {
                "status": "error",
                "message": "FAISS vector store index not found on disk.",
                "chunks": [],
                "formatted_context": "",
                "count": 0,
                "elapsed_seconds": 0.0,
            }

        vector_store = self.get_vector_store()

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🔍 [VECTOR RETRIEVAL] Searching for: '{query}' (top_k={k})")
            print("-" * 80)

        # Retrieve documents with L2 distance scores
        docs_and_scores: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(query, k=k)

        retrieved_chunks = []
        context_parts = []

        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            meta = doc.metadata
            section = meta.get("section", "General")
            component = meta.get("component", "Overview")
            p_start = meta.get("page_start", meta.get("page", "?"))
            p_end = meta.get("page_end", p_start)
            ctype = meta.get("component_type", "chunk")

            chunk_info = {
                "rank": rank,
                "score": round(float(score), 4),
                "section": section,
                "component": component,
                "component_type": ctype,
                "pages": f"P{p_start}-P{p_end}",
                "content": doc.page_content,
                "metadata": meta,
            }
            retrieved_chunks.append(chunk_info)

            # Build readable context passage
            passage = (
                f"[Document Passage {rank} | Section: {section} | Component: {component} | Pages: P{p_start}-P{p_end}]\n"
                f"{doc.page_content}"
            )
            context_parts.append(passage)

            if self.verbose:
                preview = doc.page_content.replace("\n", " ")[:100]
                print(f"  #{rank} [Distance: {score:.3f}] [{section} -> {component}] ({meta.get('pages', 'P?')})")
                print(f"     Preview: \"{preview}...\"")

        elapsed = time.time() - start_time
        formatted_context = "\n\n".join(context_parts)

        if self.verbose:
            print(f"✅ Vector Retrieval Complete: {len(retrieved_chunks)} chunks retrieved in {elapsed:.3f}s.")
            print("=" * 80 + "\n")

        return {
            "status": "success",
            "query": query,
            "chunks": retrieved_chunks,
            "formatted_context": formatted_context,
            "count": len(retrieved_chunks),
            "elapsed_seconds": round(elapsed, 3),
        }
