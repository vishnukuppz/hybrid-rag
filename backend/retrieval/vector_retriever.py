import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Retrieves relevant semantic document chunks from the FAISS Database Service via HTTP.
    Replaces direct FAISS local loading — all vector operations go through the database container.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        top_k: int = 4,
        verbose: bool = True,
    ):
        self.database_url = (database_url or os.getenv("DATABASE_URL", "http://localhost:8002")).rstrip("/")
        self.top_k = top_k
        self.verbose = verbose

    def is_available(self) -> bool:
        """Check if FAISS index is available via the database service."""
        try:
            resp = requests.get(f"{self.database_url}/api/db/status", timeout=5)
            if resp.status_code == 200:
                return resp.json().get("available", False)
        except Exception as e:
            logger.warning(f"Database service unreachable at {self.database_url}: {e}")
        return False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute similarity search via the FAISS database service HTTP API.
        Returns the same schema as the original local VectorRetriever.
        """
        k = top_k or self.top_k
        start_time = time.time()

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"🔍 [VECTOR RETRIEVAL] Querying database service: '{query}' (top_k={k})")
            print("-" * 80)

        try:
            resp = requests.post(
                f"{self.database_url}/api/db/search",
                json={"query": query, "top_k": k},
                timeout=60,
            )
            if resp.status_code == 404:
                return {
                    "status": "error",
                    "message": "FAISS index not found. Please ingest a document first.",
                    "chunks": [],
                    "formatted_context": "",
                    "count": 0,
                    "elapsed_seconds": 0.0,
                }
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Database service call failed: {e}")
            return {
                "status": "error",
                "message": f"Cannot reach database service at {self.database_url}: {str(e)}",
                "chunks": [],
                "formatted_context": "",
                "count": 0,
                "elapsed_seconds": 0.0,
            }

        raw_results = data.get("results", [])
        retrieved_chunks = []
        context_parts = []

        for item in raw_results:
            rank = item.get("rank", 0)
            section = item.get("section", "General")
            component = item.get("component", "")
            pages = item.get("pages", "P?-P?")
            content = item.get("content", "")
            score = item.get("score", 0.0)
            meta = item.get("metadata", {})

            chunk_info = {
                "rank": rank,
                "score": score,
                "section": section,
                "component": component,
                "component_type": meta.get("component_type", "chunk"),
                "pages": pages,
                "content": content,
                "metadata": meta,
            }
            retrieved_chunks.append(chunk_info)

            passage = (
                f"[Document Passage {rank} | Section: {section} | Component: {component} | Pages: {pages}]\n"
                f"{content}"
            )
            context_parts.append(passage)

            if self.verbose:
                preview = content.replace("\n", " ")[:100]
                print(f"  #{rank} [Distance: {score:.3f}] [{section} -> {component}] ({pages})")
                print(f"     Preview: \"{preview}...\"")

        elapsed = time.time() - start_time
        formatted_context = "\n\n".join(context_parts)

        if self.verbose:
            print(f"✅ Vector Retrieval Complete: {len(retrieved_chunks)} chunks in {elapsed:.3f}s.")
            print("=" * 80 + "\n")

        return {
            "status": "success",
            "query": query,
            "chunks": retrieved_chunks,
            "formatted_context": formatted_context,
            "count": len(retrieved_chunks),
            "elapsed_seconds": round(elapsed, 3),
        }
