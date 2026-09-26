import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from langchain_core.documents import Document

from common.pipeline import CommonIngestionPipeline
from vector_pipeline.embeddings import get_embedding_model
from vector_pipeline.faiss_storage import FAISSVectorStoreManager

load_dotenv()


class VectorStoragePipeline:
    """
    Vector Database Ingestion Pipeline.
    Takes chunks (from CommonIngestionPipeline or raw documents),
    generates embeddings, and stores them in FAISS.
    """

    def __init__(
        self,
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        save_directory: Union[str, Path] = "faiss_index",
        batch_size: int = 128,
        verbose: bool = True,
    ):
        self.save_directory = Path(save_directory).resolve()
        self.batch_size = batch_size
        self.verbose = verbose

        self.embeddings = get_embedding_model(
            provider=embedding_provider,
            model_name=embedding_model,
        )
        self.storage_manager = FAISSVectorStoreManager(
            embeddings=self.embeddings,
            save_directory=self.save_directory,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def run_on_chunks(
        self,
        chunks: List[Document],
        append_if_exists: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute vector storage directly on pre-split chunks from CommonIngestionPipeline.
        """
        start_time = time.time()
        if not chunks:
            print("⚠️ [VECTOR PIPELINE] No chunks provided to store.")
            return {"status": "warning", "message": "No chunks", "chunks_stored": 0}

        vector_store = self.storage_manager.create_and_save(
            chunks=chunks,
            append_if_exists=append_if_exists,
        )

        total_vectors = vector_store.index.ntotal if hasattr(vector_store, "index") else len(chunks)
        elapsed = time.time() - start_time

        summary = {
            "status": "success",
            "pipeline": "vector_pipeline",
            "chunks_stored": len(chunks),
            "total_vectors_in_store": total_vectors,
            "vector_store_directory": str(self.save_directory),
            "elapsed_seconds": round(elapsed, 2),
        }

        if self.verbose:
            print("=" * 80)
            print("🎉 [VECTOR PIPELINE COMPLETE]")
            for k, v in summary.items():
                print(f"   • {k:25s}: {v}")
            print("=" * 80 + "\n")

        return summary

    def run(
        self,
        input_path: Union[str, Path],
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        mode: str = "auto",
        append_if_exists: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the full flow: Common Loader & Splitter -> Vector Storage.
        """
        # Step 1: Common loading & splitting
        common = CommonIngestionPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode=mode,
            verbose=self.verbose,
        )
        docs, chunks = common.run(input_path)

        # Step 2: Vector embedding & storage
        result = self.run_on_chunks(chunks, append_if_exists=append_if_exists)
        result["documents_loaded"] = len(docs)
        result["input_path"] = str(Path(input_path).resolve())
        return result


def main():
    parser = argparse.ArgumentParser(description="Vector Pipeline Ingestion (FAISS)")
    parser.add_argument(
        "--input",
        "-i",
        default="docs/spotify_web_app_architecture.pdf",
        help="Path to PDF/document (default: docs/spotify_web_app_architecture.pdf)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="faiss_index",
        help="Target FAISS directory (default: faiss_index)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Chunk size (default: 600)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap (default: 100)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "section", "page"],
        default="auto",
        help="Parsing mode (default: auto)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "huggingface"],
        default="openai",
        help="Embedding provider (default: openai)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing FAISS store",
    )

    args = parser.parse_args()

    pipeline = VectorStoragePipeline(
        embedding_provider=args.provider,
        save_directory=args.output_dir,
    )
    pipeline.run(
        input_path=args.input,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        mode=args.mode,
        append_if_exists=args.append,
    )


if __name__ == "__main__":
    main()
