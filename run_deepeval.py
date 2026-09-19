import os
import sys
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import List
from dotenv import load_dotenv
from tabulate import tabulate

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.hybrid_pipeline import HybridRetrievalPipeline

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")


def run_deepeval_test_suite():
    """
    Executes DeepEval evaluation suite for the Hybrid RAG Pipeline.
    Evaluates:
      1. Answer Relevancy (Score >= 0.7)
      2. Faithfulness / Hallucination resistance (Score >= 0.7)
      3. Contextual Relevancy (Score >= 0.7)
    """
    print("=" * 80)
    print("🚀 Running DeepEval Test Suite for Hybrid RAG (Vector DB + Neo4j Graph)")
    print(f"   Evaluator Model : {MODEL_NAME}")
    print(f"   Pass Threshold  : 0.70")
    print("=" * 80)

    # Initialize DeepEval metrics
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=MODEL_NAME)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=MODEL_NAME)
    context_metric = ContextualRelevancyMetric(threshold=0.7, model=MODEL_NAME)

    # Test cases representing technical architecture queries
    test_queries = [
        "Which microservice handles playback and sessions, and what are its core tables?",
        "Explain the Data Storage Layout. What data is stored in PostgreSQL versus Redis?",
        "What are the core tables and responsibilities of the Catalog Service?",
    ]

    pipeline = HybridRetrievalPipeline(vector_top_k=4, verbose=False)
    report_rows = []

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n[Test Case {idx}/{len(test_queries)}] Query: '{query}'")
        print("-> Running Hybrid RAG retrieval pipeline (FAISS + Neo4j)...", end="", flush=True)

        t0 = time.time()
        result = pipeline.run(query)
        elapsed = time.time() - t0
        print(f" [Done in {elapsed:.2f}s]")

        actual_output = result["answer"]
        vec_chunks = result["vector_result"].get("chunks", [])
        graph_triplets = result["graph_result"].get("triplets", [])

        # Construct comprehensive retrieval context from both pipelines
        retrieval_context: List[str] = []
        for c in vec_chunks:
            retrieval_context.append(
                f"Vector Passage ({c.get('section', '')} - {c.get('component', '')}): {c.get('content', '')}"
            )
        for t in graph_triplets:
            retrieval_context.append(
                f"Graph Fact: ({t.get('source')}) -[:{t.get('relationship')}]-> ({t.get('target')})"
            )

        if not retrieval_context:
            retrieval_context = ["No vector or graph context retrieved."]

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )

        print("-> Measuring Answer Relevancy...")
        relevancy_metric.measure(test_case)

        print("-> Measuring Faithfulness (Hallucination Check)...")
        faithfulness_metric.measure(test_case)

        print("-> Measuring Contextual Relevancy...")
        context_metric.measure(test_case)

        # Record metrics in report table
        metrics = [
            ("Answer Relevancy", relevancy_metric),
            ("Faithfulness", faithfulness_metric),
            ("Contextual Relevancy", context_metric),
        ]

        for m_name, m_obj in metrics:
            status_str = "✅ PASS" if m_obj.is_successful() else "❌ FAIL"
            reason_str = (m_obj.reason[:65] + "...") if m_obj.reason else "N/A"
            report_rows.append([
                f"TC #{idx}: {query[:28]}...",
                m_name,
                f"{m_obj.score:.2f}",
                status_str,
                reason_str,
            ])

    print("\n" + "=" * 80)
    print("📊 DeepEval Evaluation Summary Report (Hybrid RAG: Vector + Graph)")
    print("=" * 80)
    headers = ["Test Case", "Evaluation Metric", "Score", "Result", "Reason"]
    print(tabulate(report_rows, headers=headers, tablefmt="grid"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_deepeval_test_suite()
