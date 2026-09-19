import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()
logger = logging.getLogger(__name__)


def get_embedding_model(
    provider: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embeddings:
    """
    Factory function to initialize and return an Embeddings instance.
    Supports 'openai' (default) and 'huggingface'.
    """
    normalized_provider = provider.strip().lower()

    if normalized_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        key = api_key or os.getenv("OPENAI_API_KEY")
        model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment or arguments. "
                "Please set OPENAI_API_KEY in your .env file."
            )

        return OpenAIEmbeddings(
            model=model,
            api_key=key,
        )

    elif normalized_provider in ["huggingface", "hf"]:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. Choose 'openai' or 'huggingface'."
        )
