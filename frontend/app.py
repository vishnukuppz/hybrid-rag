import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root and virtual environment site-packages are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

venv_site_packages = PROJECT_ROOT / ".venv" / "lib"
if venv_site_packages.exists():
    for p in venv_site_packages.glob("python*/site-packages"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Backend API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# =============================================================================
# Streamlit Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Hybrid RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean CSS Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-vec {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
        font-size: 0.78rem;
        margin-right: 0.3rem;
    }
    .badge-graph {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background-color: #f3e8ff;
        color: #7e22ce;
        font-weight: 600;
        font-size: 0.78rem;
        margin-right: 0.3rem;
    }
    .badge-time {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background-color: #ecfdf5;
        color: #047857;
        font-weight: 600;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Session State
# =============================================================================
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "ingestion"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "ingestion_logs" not in st.session_state:
    st.session_state.ingestion_logs = []

if "active_file_name" not in st.session_state:
    st.session_state.active_file_name = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "use_sample_selected" not in st.session_state:
    st.session_state.use_sample_selected = False


# =============================================================================
# Backend API Helpers
# =============================================================================
def get_backend_status() -> Dict[str, Any]:
    """Queries the FastAPI backend for FAISS and Neo4j availability."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/status", timeout=4)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"ready_for_chat": False, "faiss_db": {"available": False}, "backend_online": False}


def trigger_ingestion(
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    use_sample: bool = False,
) -> Dict[str, Any]:
    """Sends ingestion request to the FastAPI backend."""
    try:
        if use_sample:
            resp = requests.post(
                f"{BACKEND_URL}/api/ingest",
                data={"use_sample": "true"},
                timeout=180,
            )
        else:
            files = {"file": (file_name, file_bytes, "application/octet-stream")}
            resp = requests.post(
                f"{BACKEND_URL}/api/ingest",
                files=files,
                data={"use_sample": "false"},
                timeout=180,
            )

        if resp.status_code == 200:
            return resp.json()
        else:
            err_detail = resp.text
            try:
                err_detail = resp.json().get("detail", err_detail)
            except Exception:
                pass
            return {"status": "error", "message": f"API Error ({resp.status_code}): {err_detail}"}
    except Exception as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}


def send_chat_query(question: str) -> Dict[str, Any]:
    """Sends chat query to the FastAPI backend."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "question": question,
                "enable_guardrails": True,
                "top_k_vector": 4,
            },
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            err_detail = resp.text
            try:
                err_detail = resp.json().get("detail", err_detail)
            except Exception:
                pass
            return {"answer": f"⚠️ Backend error: {err_detail}", "is_refusal": False, "error": True}
    except Exception as e:
        return {"answer": f"⚠️ Could not reach Backend API at {BACKEND_URL}: {e}", "is_refusal": False, "error": True}


# Fetch backend status
api_status = get_backend_status()
faiss_ready = api_status.get("ready_for_chat", False) or api_status.get("faiss_db", {}).get("available", False)
backend_online = api_status.get("backend_online", True) if "backend_online" in api_status else True

# =============================================================================
# Sidebar (Minimal: Screen Navigation only)
# =============================================================================
with st.sidebar:
    st.title("⚡ Hybrid RAG")

    if not backend_online and not faiss_ready:
        st.error(f"⚠️ FastAPI Backend is offline at `{BACKEND_URL}`.\nRun: `python backend/main.py`")

    screen_choice = st.radio(
        "Navigation",
        ["📤 Upload & Ingestion", "💬 Chat Bot"],
        index=0 if st.session_state.current_screen == "ingestion" else 1,
        label_visibility="collapsed",
    )
    desired_screen = "ingestion" if screen_choice == "📤 Upload & Ingestion" else "chat"
    if desired_screen != st.session_state.current_screen:
        st.session_state.current_screen = desired_screen
        st.rerun()

    if st.session_state.current_screen == "chat":
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()


# =============================================================================
# SCREEN 1: FILE UPLOAD & INGESTION LOGS
# =============================================================================
if st.session_state.current_screen == "ingestion":
    st.markdown('<div class="main-title">📤 Upload Document & Ingestion</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a document to process it through the FastAPI backend into the FAISS Vector DB and Neo4j Knowledge Graph.</div>',
        unsafe_allow_html=True,
    )

    # 1. Upload section
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Document (PDF, TXT, MD, JSON)",
            type=["pdf", "txt", "md", "json"],
            help="Select any document file to ingest.",
        )
    with col_sample:
        st.write("")
        st.write("")
        if st.button("📄 Load Spotify Architecture PDF", use_container_width=True):
            st.session_state.use_sample_selected = True
            st.session_state.active_file_name = "spotify_web_app_architecture.pdf"

    if uploaded_file is not None:
        st.session_state.active_file_name = uploaded_file.name
        st.session_state.use_sample_selected = False
        st.info(f"📁 Selected File: **{uploaded_file.name}** ({len(uploaded_file.getvalue()) / 1024.0:.2f} KB)")
    elif st.session_state.use_sample_selected:
        st.info("📁 Sample File Selected: **spotify_web_app_architecture.pdf**")

    # 2. Action buttons
    doc_is_uploaded = uploaded_file is not None or st.session_state.use_sample_selected or st.session_state.active_file_name is not None

    st.write("")
    col_btn_ingest, col_btn_chat = st.columns(2)

    with col_btn_ingest:
        start_ingest = st.button(
            "🚀 Ingest Document",
            type="primary" if not faiss_ready else "secondary",
            disabled=(not doc_is_uploaded),
            use_container_width=True,
            help="Sends document to FastAPI backend to process through Vector & Graph pipelines.",
        )

    # Move to Chat button requirement:
    # Visible once the doc is uploaded/selected.
    # Enabled only if FAISS DB is available on disk, otherwise greyed out.
    with col_btn_chat:
        if doc_is_uploaded:
            if faiss_ready:
                if st.button("💬 Move to Chat ➡️", type="primary", use_container_width=True):
                    st.session_state.current_screen = "chat"
                    st.rerun()
            else:
                st.button(
                    "💬 Move to Chat ➡️",
                    disabled=True,
                    use_container_width=True,
                    help="FAISS database not found. Ingest the document first to enable chat.",
                )
                st.caption("🔒 *FAISS Vector Database is not built yet. Click 'Ingest Document' first.*")

    # 3. Processing & Logging via FastAPI
    if start_ingest:
        status_box = st.status("Ingesting document via FastAPI Backend...", expanded=True)
        t_start = time.time()

        status_box.write("📡 Sending document to FastAPI backend at `/api/ingest`...")

        if st.session_state.use_sample_selected:
            result = trigger_ingestion(use_sample=True)
        elif uploaded_file is not None:
            result = trigger_ingestion(
                file_bytes=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
                use_sample=False,
            )
        else:
            result = trigger_ingestion(use_sample=True)

        if result.get("status") == "success":
            logs = result.get("logs", [])
            st.session_state.ingestion_logs = logs
            for log_line in logs:
                status_box.write(log_line)
            t_total = time.time() - t_start
            status_box.update(label=f"🎉 Ingestion Complete ({t_total:.2f}s)!", state="complete", expanded=False)
            st.toast("Ingestion complete! You can now click 'Move to Chat'.", icon="✅")
            time.sleep(1)
            st.rerun()
        else:
            err_msg = result.get("message", "Unknown ingestion error occurred.")
            status_box.update(label=f"❌ Error: {err_msg}", state="error")
            st.error(f"Ingestion failed: {err_msg}")

    # 4. Logs of what happened container
    if st.session_state.ingestion_logs:
        st.markdown("---")
        st.markdown("### 📋 Ingestion Activity Logs")
        with st.container():
            for log_entry in st.session_state.ingestion_logs:
                st.markdown(f"• {log_entry}")


# =============================================================================
# SCREEN 2: CHAT BOT
# =============================================================================
elif st.session_state.current_screen == "chat":
    # Header bar
    col_back, col_h = st.columns([1.5, 6])
    with col_back:
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.current_screen = "ingestion"
            st.rerun()
    with col_h:
        active_name = st.session_state.active_file_name or "Ingested Document"
        st.markdown('<div class="main-title" style="font-size:1.7rem;">💬 Hybrid RAG Chatbot</div>', unsafe_allow_html=True)
        st.caption(f"Knowledge Base: **{active_name}** | Dual Retrieval: **FAISS Vector DB + Neo4j Graph (via FastAPI)**")

    # Suggested Questions
    st.markdown("##### 💡 Suggested Questions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🎵 Playback & Sessions", use_container_width=True):
            st.session_state.pending_prompt = "Which microservice handles playback and sessions, and what are its core tables?"
    with c2:
        if st.button("🗄️ PostgreSQL vs Redis", use_container_width=True):
            st.session_state.pending_prompt = "Explain the Data Storage Layout. What data is stored in PostgreSQL versus Redis?"
    with c3:
        if st.button("📑 Catalog Service Tables", use_container_width=True):
            st.session_state.pending_prompt = "What are the core tables and responsibilities of the Catalog Service?"
    with c4:
        if st.button("🌐 Frontend Routes", use_container_width=True):
            st.session_state.pending_prompt = "What routes are available in the authenticated Next.js frontend structure?"

    st.markdown("---")

    # Render chat messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("details"):
                det = msg["details"]
                v_chunks = det.get("vector_chunks", [])
                g_triplets = det.get("graph_triplets", [])
                t_sec = det.get("timings", {}).get("total_seconds", 0)

                st.markdown(
                    f"""
                    <span class="badge-vec">🔍 Vector Passages: {len(v_chunks)}</span>
                    <span class="badge-graph">🕸️ Graph Triplets: {len(g_triplets)}</span>
                    <span class="badge-time">⏱️ {t_sec:.2f}s</span>
                    """,
                    unsafe_allow_html=True,
                )

                if v_chunks or g_triplets:
                    with st.expander("🔍 View Retrieval Evidence (Vector Passages & Graph Facts)"):
                        if v_chunks:
                            st.markdown("**Vector DB Evidence:**")
                            for c in v_chunks:
                                section = c.get("section", "")
                                comp = c.get("component", "")
                                pages = c.get("pages", "")
                                header_str = f"• **{section}"
                                if comp:
                                    header_str += f" -> {comp}"
                                header_str += f"** ({pages}):"
                                st.markdown(header_str)
                                st.caption(c.get("content", ""))
                        if g_triplets:
                            st.markdown("**Knowledge Graph Evidence:**")
                            for t in g_triplets:
                                st.markdown(f"• `({t.get('source')}) -[:{t.get('relationship')}]-> ({t.get('target')})`")

    # Chat Input
    user_prompt = st.chat_input("Ask a question about the document...")
    if st.session_state.pending_prompt:
        user_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Querying Hybrid RAG via FastAPI Backend..."):
                resp_data = send_chat_query(user_prompt)

                if resp_data.get("is_refusal"):
                    refusal = resp_data.get("answer") or "Query intercepted by Guardrail."
                    category = resp_data.get("guardrail_status", "blocked")
                    st.warning(f"🛡️ **Guardrail Intercepted ({category})**\n\n{refusal}")
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": refusal,
                        "details": None,
                    })
                elif resp_data.get("error"):
                    st.error(resp_data.get("answer"))
                else:
                    answer = resp_data.get("answer", "")
                    st.markdown(answer)

                    v_evidence = resp_data.get("vector_evidence", [])
                    g_evidence = resp_data.get("graph_evidence", {})
                    timings = resp_data.get("timing", {})
                    g_triplets = g_evidence.get("triplets", [])
                    t_sec = timings.get("total_seconds", 0)

                    st.markdown(
                        f"""
                        <span class="badge-vec">🔍 Vector Passages: {len(v_evidence)}</span>
                        <span class="badge-graph">🕸️ Graph Triplets: {len(g_triplets)}</span>
                        <span class="badge-time">⏱️ {t_sec:.2f}s</span>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("🔍 View Retrieval Evidence (Vector Passages & Graph Facts)"):
                        if v_evidence:
                            st.markdown("**Vector DB Evidence:**")
                            for c in v_evidence:
                                section = c.get("section", "")
                                comp = c.get("component", "")
                                pages = c.get("pages", "")
                                header_str = f"• **{section}"
                                if comp:
                                    header_str += f" -> {comp}"
                                header_str += f"** ({pages}):"
                                st.markdown(header_str)
                                st.caption(c.get("content", ""))
                        if g_triplets:
                            st.markdown("**Knowledge Graph Evidence:**")
                            for t in g_triplets:
                                st.markdown(f"• `({t.get('source')}) -[:{t.get('relationship')}]-> ({t.get('target')})`")

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "details": {
                            "vector_chunks": v_evidence,
                            "graph_triplets": g_triplets,
                            "timings": timings,
                        },
                    })
