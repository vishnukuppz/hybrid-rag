import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from langchain_core.documents import Document

from common.pipeline import CommonIngestionPipeline
from graph_pipeline.entity_extractor import EntityExtractor
from graph_pipeline.neo4j_storage import Neo4jStorageManager

load_dotenv()


class GraphStoragePipeline:
    """
    Knowledge Graph (K-RAG) Pipeline for Neo4j.
    Takes chunks (from CommonIngestionPipeline or raw documents),
    extracts entities & relationships using LLMGraphTransformer,
    and stores them into Neo4j.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_concurrency: int = 5,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.extractor = EntityExtractor(
            model_name=model_name,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            max_concurrency=max_concurrency,
            verbose=self.verbose,
        )
        self.storage = Neo4jStorageManager(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            verbose=self.verbose,
        )

    def run_on_chunks(
        self,
        chunks: List[Document],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute graph extraction and Neo4j storage directly on pre-split chunks.
        """
        start_time = time.time()
        if not chunks:
            print("⚠️ [GRAPH PIPELINE] No chunks provided to process.")
            return {"status": "warning", "message": "No chunks", "graph_documents_persisted": 0}

        # Step 1: Check Neo4j connection upfront
        if not self.storage.test_connection():
            raise ConnectionError("Neo4j database is unreachable. Check credentials in .env.")

        # Step 2: Extract entity graphs
        graph_docs = self.extractor.extract_graph_documents(
            chunks,
            show_progress=show_progress,
        )

        # Step 3: Persist to Neo4j
        persisted_count = self.storage.add_graph_documents(
            graph_docs,
            base_entity_label=True,
            include_source=True,
        )

        stats = self.storage.get_graph_stats()
        elapsed = time.time() - start_time

        summary = {
            "status": "success",
            "pipeline": "graph_pipeline",
            "chunks_processed": len(chunks),
            "graph_documents_extracted": len(graph_docs),
            "graph_documents_persisted": persisted_count,
            "neo4j_total_nodes": stats["node_count"],
            "neo4j_total_relationships": stats["relationship_count"],
            "elapsed_seconds": round(elapsed, 2),
        }

        if self.verbose:
            print("=" * 80)
            print("🎉 [GRAPH PIPELINE COMPLETE]")
            for k, v in summary.items():
                print(f"   • {k:28s}: {v}")
            print("=" * 80 + "\n")

        return summary

    def run(
        self,
        input_path: Union[str, Path],
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        mode: str = "auto",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the full flow: Common Loader & Splitter -> Graph Storage.
        """
        # Step 1: Common loading & splitting
        common = CommonIngestionPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode=mode,
            verbose=self.verbose,
        )
        docs, chunks = common.run(input_path)

        # Step 2: Graph extraction & storage
        result = self.run_on_chunks(chunks, show_progress=show_progress)
        result["documents_loaded"] = len(docs)
        result["input_path"] = str(Path(input_path).resolve())
        return result


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Pipeline Ingestion (Neo4j)")
    parser.add_argument(
        "--input",
        "-i",
        default="docs/spotify_web_app_architecture.pdf",
        help="Path to document (default: docs/spotify_web_app_architecture.pdf)",
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
        default=80,
        help="Chunk overlap (default: 80)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrency limit (default: 5)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "section", "page"],
        default="auto",
        help="Parsing mode (default: auto)",
    )

    args = parser.parse_args()

    pipeline = GraphStoragePipeline(
        model_name=args.model,
        max_concurrency=args.concurrency,
    )
    pipeline.run(
        input_path=args.input,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
