import logging
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval.vector_retriever import VectorRetriever
from retrieval.graph_retriever import GraphRetriever

load_dotenv()
logger = logging.getLogger(__name__)

HYBRID_RAG_PROMPT = """You are an advanced AI assistant powered by a Hybrid RAG architecture combining Dense Vector Retrieval (document text chunks) and Knowledge Graph Retrieval (Neo4j entity-relationship graph).

Your goal is to answer the user's question accurately, comprehensively, and clearly by synthesizing information from BOTH retrieval sources.

User Question:
{question}

================================================================================
SOURCE 1: VECTOR DATABASE CONTEXT (Document Passages & Sections)
================================================================================
{vector_context}

================================================================================
SOURCE 2: KNOWLEDGE GRAPH CONTEXT (Neo4j Entities & Relationships)
================================================================================
{graph_context}

================================================================================
ANSWER INSTRUCTIONS:
1. Ground your answer thoroughly in the provided Vector and Graph contexts.
2. Structure your answer with clear headings, bullet points, and highlight relevant components, microservices, database tables, or technologies.
3. When referencing specific components, cite their section, service number, or connected relationships where applicable.
4. If there are contradictions or complementary details between text passages and graph relationships, integrate them coherently.
5. If the context does not contain sufficient details to answer the question, clearly state what is known from the context and what is missing.
"""


class HybridRetrievalPipeline:
    """
    Master Hybrid RAG Retrieval Pipeline.
    Retrieves from:
      - FAISS Database Service (via HTTP API)
      - Neo4j Knowledge Graph (entity/relationship triplets)
    Fuses both contexts and synthesises the answer with an LLM.
    """

    def __init__(
        self,
        vector_top_k: int = 4,
        llm_model: Optional[str] = None,
        database_url: Optional[str] = None,
        verbose: bool = True,
    ):
        self.vector_top_k = vector_top_k
        self.verbose = verbose
        self.llm_model = llm_model or os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")
        self.database_url = database_url or os.getenv("DATABASE_URL", "http://localhost:8002")

        self.vector_retriever = VectorRetriever(
            database_url=self.database_url,
            top_k=self.vector_top_k,
            verbose=self.verbose,
        )
        self.graph_retriever = GraphRetriever(
            model_name=self.llm_model,
            verbose=self.verbose,
        )
        self.synthesizer_llm = ChatOpenAI(model_name=self.llm_model, temperature=0.2)
        self.prompt = PromptTemplate.from_template(HYBRID_RAG_PROMPT)
        self.chain = self.prompt | self.synthesizer_llm | StrOutputParser()

    def run(self, question: str) -> Dict[str, Any]:
        """Execute Hybrid Retrieval & Synthesis."""
        overall_start = time.time()

        if self.verbose:
            print("\n" + "█" * 80)
            print("  ⚡ HYBRID RAG RETRIEVAL PIPELINE")
            print("█" * 80)
            print(f"  User Query: \"{question}\"")
            print("█" * 80)

        # 1. Vector retrieval via database service
        t_vec_start = time.time()
        vector_res = self.vector_retriever.retrieve(question, top_k=self.vector_top_k)
        t_vec = time.time() - t_vec_start

        # 2. Graph retrieval via Neo4j
        t_graph_start = time.time()
        graph_res = self.graph_retriever.retrieve(question)
        t_graph = time.time() - t_graph_start

        # 3. Context fusion
        vector_context = vector_res.get("formatted_context", "").strip() or "No relevant passages found."
        graph_context = graph_res.get("formatted_context", "").strip() or "No relevant entities or relationships found."

        if self.verbose:
            print("\n" + "=" * 80)
            print("🤖 [HYBRID SYNTHESIS] Generating Answer with LLM...")
            print("-" * 80)

        t_synth_start = time.time()
        answer = self.chain.invoke({
            "question": question,
            "vector_context": vector_context,
            "graph_context": graph_context,
        })
        t_synth = time.time() - t_synth_start
        overall_time = time.time() - overall_start

        if self.verbose:
            print("\n" + "=" * 80)
            print("🏆 FINAL HYBRID RAG ANSWER:")
            print("=" * 80)
            print(answer)
            print("=" * 80)
            print(f"⏱️ Timings: Vector={t_vec:.2f}s | Graph={t_graph:.2f}s | Synthesis={t_synth:.2f}s | Total={overall_time:.2f}s\n")

        return {
            "question": question,
            "answer": answer,
            "vector_result": vector_res,
            "graph_result": graph_res,
            "timings": {
                "vector_retrieval_seconds": round(t_vec, 3),
                "graph_retrieval_seconds": round(t_graph, 3),
                "synthesis_seconds": round(t_synth, 3),
                "total_seconds": round(overall_time, 3),
            },
        }
