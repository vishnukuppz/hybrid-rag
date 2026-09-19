import asyncio
import concurrent.futures
import logging
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts knowledge graph entities and relationships from document chunks
    using LangChain's LLMGraphTransformer and OpenAI Chat models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        allowed_nodes: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
        max_concurrency: int = 5,
        verbose: bool = True,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")
        self.max_concurrency = max_concurrency
        self.verbose = verbose

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )

        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=self.model_name,
            api_key=self.api_key,
        )

        transformer_kwargs = {"llm": self.llm}
        if allowed_nodes:
            transformer_kwargs["allowed_nodes"] = allowed_nodes
        if allowed_relationships:
            transformer_kwargs["allowed_relationships"] = allowed_relationships

        self.transformer = LLMGraphTransformer(**transformer_kwargs)

    async def _extract_async(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> List[GraphDocument]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        total_chunks = len(documents)

        if self.verbose:
            print("\n" + "=" * 80)
            print("🤖 [GRAPH PIPELINE: ENTITY & RELATIONSHIP EXTRACTION]")
            print(f"   🧠 Model: {self.model_name}")
            print(f"   ⚡ Concurrency: {self.max_concurrency} parallel requests")
            print(f"   📦 Chunks to Process: {total_chunks}")
            print("-" * 80)

        processed_counter = 0

        async def _process_single_doc(chunk_idx: int, doc: Document, pbar: Optional[tqdm]):
            nonlocal processed_counter
            async with semaphore:
                comp = doc.metadata.get("component") or doc.metadata.get("section") or f"Chunk #{chunk_idx}"
                t0 = time.time()
                try:
                    res: GraphDocument = await self.transformer.aprocess_response(doc)
                    elapsed = time.time() - t0
                    processed_counter += 1

                    if self.verbose and res:
                        node_count = len(res.nodes)
                        rel_count = len(res.relationships)
                        sample_nodes = ", ".join([f"{n.id} ({n.type})" for n in res.nodes[:3]])
                        if len(res.nodes) > 3:
                            sample_nodes += f" ... (+{len(res.nodes)-3} more)"

                        sample_rels = ", ".join([f"({r.source.id})-[{r.type}]->({r.target.id})" for r in res.relationships[:2]])
                        if len(res.relationships) > 2:
                            sample_rels += f" ... (+{len(res.relationships)-2} more)"

                        print(
                            f"  [{processed_counter:02d}/{total_chunks:02d}] {comp[:35]:35s} "
                            f"({elapsed:.1f}s) -> 🟢 {node_count:2d} nodes, 🔗 {rel_count:2d} rels"
                        )
                        if sample_nodes:
                            print(f"       Nodes: {sample_nodes}")
                        if sample_rels:
                            print(f"       Rels : {sample_rels}")
                    return res

                except Exception as e:
                    print(f"  ❌ [{chunk_idx:02d}/{total_chunks:02d}] Extraction failed on {comp}: {e}")
                    return None
                finally:
                    if pbar:
                        pbar.update(1)

        pbar = (
            tqdm(total=total_chunks, desc="LLM Graph Extraction", unit="chunk")
            if (show_progress and not self.verbose)
            else None
        )

        try:
            tasks = [_process_single_doc(idx, doc, pbar) for idx, doc in enumerate(documents, start=1)]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]

            total_nodes = sum(len(g.nodes) for g in valid_results)
            total_rels = sum(len(g.relationships) for g in valid_results)

            if self.verbose:
                print("-" * 80)
                print(f"✅ Extraction Complete: {len(valid_results)}/{total_chunks} graph documents extracted.")
                print(f"📊 Totals: 🟢 {total_nodes} nodes | 🔗 {total_rels} relationships.")
                print("=" * 80 + "\n")

            return valid_results
        finally:
            if pbar:
                pbar.close()

    def extract_graph_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> List[GraphDocument]:
        if not documents:
            if self.verbose:
                print("⚠️ No documents provided for entity extraction.")
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self._extract_async(documents, show_progress=show_progress),
                ).result()
        else:
            return asyncio.run(
                self._extract_async(documents, show_progress=show_progress)
            )
