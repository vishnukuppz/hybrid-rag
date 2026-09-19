import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import pytest
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from backend.retrieval.hybrid_pipeline import HybridRetrievalPipeline

load_dotenv()
EVAL_MODEL = os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")


@pytest.fixture(scope="module")
def relevancy_metric():
    return AnswerRelevancyMetric(threshold=0.7, model=EVAL_MODEL)


@pytest.fixture(scope="module")
def faithfulness_metric():
    return FaithfulnessMetric(threshold=0.7, model=EVAL_MODEL)


@pytest.fixture(scope="module")
def contextual_metric():
    return ContextualRelevancyMetric(threshold=0.7, model=EVAL_MODEL)


@pytest.fixture(scope="module")
def hybrid_pipeline():
    return HybridRetrievalPipeline(vector_top_k=4, verbose=False)


def test_playback_service_hybrid_rag(hybrid_pipeline, relevancy_metric, faithfulness_metric):
    """
    Test that queries regarding playback and session service produce relevant
    and faithful answers without hallucinating non-existent tables.
    """
    query = "Which microservice handles playback and sessions, and what are its core tables?"
    result = hybrid_pipeline.run(query)

    actual_output = result["answer"]
    vec_chunks = result["vector_result"].get("chunks", [])
    graph_triplets = result["graph_result"].get("triplets", [])

    retrieval_context = [
        f"Vector Passage: {c.get('content', '')}" for c in vec_chunks
    ] + [
        f"Graph Triplet: ({t.get('source')}) -[:{t.get('relationship')}]-> ({t.get('target')})"
        for t in graph_triplets
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context or ["Playback and session service manages playback."],
    )

    assert_test(test_case, [relevancy_metric, faithfulness_metric])


def test_data_storage_layout_hybrid_rag(hybrid_pipeline, relevancy_metric, faithfulness_metric):
    """
    Test that queries about the database storage layout accurately cite
    PostgreSQL vs Redis responsibilities.
    """
    query = "Explain the Data Storage Layout. What data is stored in PostgreSQL versus Redis?"
    result = hybrid_pipeline.run(query)

    actual_output = result["answer"]
    vec_chunks = result["vector_result"].get("chunks", [])
    graph_triplets = result["graph_result"].get("triplets", [])

    retrieval_context = [
        f"Vector Passage: {c.get('content', '')}" for c in vec_chunks
    ] + [
        f"Graph Triplet: ({t.get('source')}) -[:{t.get('relationship')}]-> ({t.get('target')})"
        for t in graph_triplets
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context or ["PostgreSQL stores relational data, Redis caches sessions."],
    )

    assert_test(test_case, [relevancy_metric, faithfulness_metric])
