# Hybrid RAG Architecture & Ingestion System

A modular Hybrid Retrieval-Augmented Generation (Hybrid RAG) architecture featuring a **shared preprocessing pipeline** (Document Loader & Text Splitter) that feeds into **two dedicated downstream pipelines**:
1. **Vector Storage Pipeline** (Dense Semantic Embeddings + FAISS)
2. **Knowledge Graph Pipeline** (LLM Entity/Relation Extraction + Neo4j)

---

### Directory Structure

```
hybrid_rag/
├── .env                       # Credentials (OpenAI API key, Neo4j connection)
├── requirements.txt           # Project dependencies
│
├── frontend/                  # 🎨 FRONTEND (Streamlit UI only)
│   └── app.py                 # Clean interactive Web UI & Hybrid Chatbot
│
├── backend/                   # ⚙️ BACKEND (All Ingestion & Retrieval Pipelines)
│   ├── __init__.py            # Clean exports for all pipeline classes
│   ├── ingest.py              # Master Ingestion CLI (Unified coordinator)
│   ├── common/                # 🔶 STAGE 1: COMMON PIPELINE
│   │   ├── __init__.py
│   │   ├── document_loader.py # Architecture-aware loader (PDF, TXT, JSON)
│   │   ├── text_splitter.py   # Semantic text splitter with metadata enrichment
│   │   └── pipeline.py        # CommonIngestionPipeline (Loads & chunks once)
│   │
│   ├── vector_pipeline/       # 🔷 STAGE 2A: VECTOR PIPELINE
│   │   ├── __init__.py
│   │   ├── embeddings.py      # OpenAI (text-embedding-3-small) / HuggingFace
│   │   ├── faiss_storage.py   # FAISS index manager & local persistence
│   │   └── pipeline.py        # VectorStoragePipeline (accepts common chunks)
│   │
│   ├── graph_pipeline/        # 🔷 STAGE 2B: GRAPH (K-RAG) PIPELINE
│   │   ├── __init__.py
│   │   ├── entity_extractor.py# LLMGraphTransformer entity/relation extractor
│   │   ├── neo4j_storage.py   # Neo4j graph manager (connection & storage)
│   │   └── pipeline.py        # GraphStoragePipeline (accepts common chunks)
│   │
│   ├── retrieval/             # 🔷 RETRIEVAL & FUSION PIPELINE
│   │   ├── __init__.py
│   │   ├── vector_retriever.py# FAISS vector similarity search
│   │   ├── graph_retriever.py # Cypher query generation & neighborhood search
│   │   └── hybrid_pipeline.py # Context fusion & LLM synthesis
│   │
│   ├── guardrails/            # 🛡️ GUARDRAILS & MODERATION
│   │   ├── __init__.py
│   │   └── manager.py         # NeMo Guardrails + Structured LLM Evaluator
│   │
│   └── guardrails_config/     # Colang & YAML Guardrail policies
│       ├── config.yml
│       └── rails.co
│
├── docs/                      # Sample documents (Spotify architecture PDF)
├── faiss_index/               # Local FAISS vector index files
├── tests/                     # DeepEval & PyTest test cases
└── run_deepeval.py            # DeepEval automated benchmark runner
```

---

## How It Works

```
                     ┌─────────────────────────────────────────┐
                     │ docs/spotify_web_app_architecture.pdf   │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │   [backend/common/document_loader.py]   │
                     │  - Detects architectural specifications │
                     │  - Stitches cross-page services/routes  │
                     │  - Extracts 46 atomic semantic units    │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │    [backend/common/text_splitter.py]    │
                     │  - Preserves section hierarchy metadata │
                     │  - Generates 48 contextual chunks       │
                     └────────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
                  ▼                                               ▼
   ┌───────────────────────────────┐               ┌───────────────────────────────┐
   │    backend/vector_pipeline/   │               │    backend/graph_pipeline/    │
   │  - OpenAI Embeddings          │               │  - LLMGraphTransformer       │
   │  - FAISS Vector DB Index      │               │  - Concurrent extraction      │
   │  - Writes index.faiss & .pkl  │               │  - Persists to Neo4j Graph    │
   └───────────────────────────────┘               └───────────────────────────────┘
```

---

## Quickstart: Running Backend & Frontend

### 1. Start the FastAPI Backend Server
The FastAPI backend coordinates all pipeline tasks (document ingestion, FAISS indexing, Neo4j graph extraction, hybrid retrieval, and safety guardrails):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Status Endpoint**: [http://localhost:8000/api/status](http://localhost:8000/api/status)

### 2. Launch the Streamlit Frontend
The frontend connects purely via HTTP requests to the FastAPI backend:

```bash
streamlit run frontend/app.py
```

---

## Streamlit Studio & Hybrid Chat Interface

### Screen Flow & Features:
1. **Screen 1: Document Ingestion**
   - **Upload Document**: Upload any PDF, TXT, MD, or JSON file directly from the browser (or click *"Load Spotify Architecture PDF"*).
   - **Ingest Document**: Dispatches upload to FastAPI `/api/ingest` which runs the common loading and splitting, FAISS embedding, and Neo4j knowledge graph storage.
   - **Move to Chat Button**:
     - Once a document is selected or uploaded, the **"💬 Move to Hybrid RAG Chat ➡️"** button becomes visible.
     - If the FAISS database is available on disk, the button is **enabled** (`primary`).
     - If FAISS is not yet built, the button is **greyed out / disabled** with a helpful tooltip prompting the user to ingest the document first.
2. **Screen 2: Hybrid RAG Chat**
   - **Dual Retrieval Execution**: For every question, concurrently retrieves from:
     - 🔍 **FAISS Vector DB**: Top-k semantically similar text passages with metadata, page numbers, and distance scores.
     - 🕸️ **Neo4j Knowledge Graph**: Entity & intent extraction, candidate matching, generated Cypher query, and multi-hop graph triplets.
   - **Context Fusion & Synthesis**: Uses `gpt-4o-mini` to synthesize a grounded answer integrating both text passages and graph relationships.
   - **Interactive Diagnostics**: Collapsible inspection drawers for Vector DB passages, Cypher query, Graph triplets table, and retrieval timing breakdown.
   - **Sample Questions**: 1-click suggested prompts tailored to the architecture document.

---

## Safety & Guardrails (`guardrails/`)

The system includes a dual-layer Guardrail architecture based on **NVIDIA NeMo Guardrails** (`guardrails_config/`) and an LLM Security & Domain Moderation Evaluator:

1. **Input Moderation**:
   - **Off-topic Query Detection**: Intercepts casual chitchat, politics, sports, recipes, or finance and returns a domain-specific refusal.
   - **Jailbreak & Prompt Injection Defense**: Intercepts DAN attempts, system prompt leaks, and policy overrides.
2. **Streamlit Integration**:
   - Includes a **"🛡️ Enable Guardrails"** toggle in the sidebar.
   - When triggered, intercepted queries display an alert card without wasting API tokens or querying databases.

Test Guardrails directly:

```bash
python -c "
from guardrails.manager import HybridRAGGuardrailManager
gm = HybridRAGGuardrailManager()
print(gm.evaluate('What is the stock price of Apple?'))
"
```

---

## DeepEval Automated Evaluation Suite

The system includes automated RAG evaluation metrics powered by **DeepEval**:
- **Answer Relevancy** (`threshold=0.7`): Verifies the answer directly answers the query.
- **Faithfulness** (`threshold=0.7`): Checks for factual hallucinations against combined Vector + Graph context.
- **Contextual Relevancy** (`threshold=0.7`): Measures retrieval precision.

### 1. Run Interactive CLI Report

```bash
python run_deepeval.py
```

### 2. Run Pytest Suite

```bash
pytest tests/test_hybrid_rag_deepeval.py -v
```

---

## Standalone Retrieval CLI

### 1. Unified Master Ingestion (`ingest.py`)

Run the common pipeline and dispatch to **both** or **either** pipeline:

```bash
# Ingest into BOTH Vector (FAISS) and Graph (Neo4j) pipelines
python backend/ingest.py --target both --input docs/spotify_web_app_architecture.pdf

# Ingest into Vector Pipeline only
python backend/ingest.py --target vector --input docs/spotify_web_app_architecture.pdf

# Ingest into Graph Pipeline only
python backend/ingest.py --target graph --input docs/spotify_web_app_architecture.pdf
```

### 2. Standalone Pipeline Execution

Each downstream pipeline can also be run independently:

```bash
# Run Vector Pipeline directly
python -m backend.vector_pipeline.pipeline --input docs/spotify_web_app_architecture.pdf

# Run Knowledge Graph Pipeline directly
python -m backend.graph_pipeline.pipeline --input docs/spotify_web_app_architecture.pdf --concurrency 5
```

---

## Python API Usage

```python
from backend import (
    CommonIngestionPipeline,
    VectorStoragePipeline,
    GraphStoragePipeline,
    HybridRetrievalPipeline,
)

# Step 1: Run common loading & splitting ONCE
common = CommonIngestionPipeline(chunk_size=600, chunk_overlap=100)
documents, chunks = common.run("docs/spotify_web_app_architecture.pdf")

# Step 2A: Feed chunks into Vector Pipeline
vector_pipe = VectorStoragePipeline(save_directory="faiss_index")
vector_result = vector_pipe.run_on_chunks(chunks)

# Step 2B: Feed the exact same chunks into Knowledge Graph Pipeline
graph_pipe = GraphStoragePipeline(model_name="gpt-4o-mini", max_concurrency=5)
graph_result = graph_pipe.run_on_chunks(chunks)
```
