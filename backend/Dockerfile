# =============================================================================
# Backend Dockerfile - FastAPI + Hybrid RAG Pipeline
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Ensure unbuffered logs & bytecode exclusion
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies & curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source code & default documentation
COPY backend/ /app/backend/
COPY docs/ /app/docs/

# Create persistent storage folder for FAISS index
RUN mkdir -p /app/faiss_index

EXPOSE 8000

# Health check to ensure FastAPI is up and responsive
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
