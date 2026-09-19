import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union
from langchain_core.documents import Document

from backend.common.document_loader import DocumentLoader
from backend.common.text_splitter import CommonTextSplitter

logger = logging.getLogger(__name__)


class CommonIngestionPipeline:
    """
    Common Preprocessing Stage for Hybrid RAG:
      1. Loads source files (PDF, TXT, JSON) with semantic section stitching
      2. Splits into contextual, enriched chunks with metadata
    Returns the loaded documents and chunked documents ready for downstream
    vector or graph pipelines.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        mode: str = "auto",
        verbose: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode
        self.verbose = verbose

        self.loader = DocumentLoader(mode=self.mode, verbose=self.verbose)
        self.splitter = CommonTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            verbose=self.verbose,
        )

    def run(self, input_path: Union[str, Path]) -> Tuple[List[Document], List[Document]]:
        """
        Execute the common loading and splitting pipeline.
        Returns: (documents, chunks)
        """
        path = Path(input_path).resolve()
        start_time = time.time()

        if self.verbose:
            print("\n" + "═" * 80)
            print("  🔷 HYBRID RAG: COMMON INGESTION PIPELINE (LOAD & SPLIT)")
            print("═" * 80)
            print(f"  Target Source       : {path}")
            print(f"  Parsing Mode        : {self.mode}")
            print(f"  Chunking Parameters : chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
            print("═" * 80)

        # Stage 1: Document Loading
        documents = self.loader.load(path)

        if not documents:
            if self.verbose:
                print("❌ [COMMON PIPELINE] No valid documents extracted.")
            return [], []

        # Stage 2: Chunking & Metadata Enrichment
        chunks = self.splitter.split_documents(documents)

        elapsed = time.time() - start_time
        if self.verbose:
            print(
                f"🎉 [COMMON PIPELINE COMPLETE] Prepared {len(chunks)} chunks "
                f"from {len(documents)} document units in {elapsed:.2f}s.\n"
            )

        return documents, chunks
