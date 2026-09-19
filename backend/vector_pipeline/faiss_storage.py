import logging
import time
from pathlib import Path
from typing import List, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FAISSVectorStoreManager:
    """
    Manages building, batch-inserting, and persisting FAISS vector databases
    with real-time terminal feedback.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        save_directory: Union[str, Path] = "faiss_index",
        batch_size: int = 128,
        verbose: bool = True,
    ):
        self.embeddings = embeddings
        self.save_directory = Path(save_directory).resolve()
        self.batch_size = batch_size
        self.verbose = verbose

    def create_and_save(
        self,
        chunks: List[Document],
        append_if_exists: bool = False,
    ) -> FAISS:
        if not chunks:
            raise ValueError("No document chunks provided to store.")

        total_chunks = len(chunks)

        if self.verbose:
            print("\n" + "=" * 80)
            print("🧠 [VECTOR PIPELINE: EMBEDDING & FAISS STORAGE]")
            print(f"   📦 Chunks to Embed: {total_chunks}")
            print(f"   ⚙️ Batch Size: {self.batch_size}")
            print(f"   💾 Destination Directory: {self.save_directory}")
            print("-" * 80)

        vector_store: Optional[FAISS] = None

        if append_if_exists and self.exists():
            if self.verbose:
                print(f"🔄 Appending to existing FAISS index at: {self.save_directory}")
            vector_store = self.load()

        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        start_all = time.time()

        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            if self.verbose:
                print(f"   ⚡ Processing Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end="", flush=True)

            t0 = time.time()
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                vector_store.add_documents(batch)
            elapsed = time.time() - t0

            if self.verbose:
                print(f" [OK] ({elapsed:.2f}s)")

        total_elapsed = time.time() - start_all
        if self.verbose:
            print(f"✅ All {total_batches} batch(es) embedded in {total_elapsed:.2f}s.")

        # Persist index to disk
        self.save_directory.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(self.save_directory))

        index_file = self.save_directory / "index.faiss"
        pkl_file = self.save_directory / "index.pkl"
        faiss_size_kb = index_file.stat().st_size / 1024.0 if index_file.exists() else 0
        pkl_size_kb = pkl_file.stat().st_size / 1024.0 if pkl_file.exists() else 0
        total_vectors = vector_store.index.ntotal if hasattr(vector_store, "index") else total_chunks
        dimension = vector_store.index.d if hasattr(vector_store, "index") else "N/A"

        if self.verbose:
            print("-" * 80)
            print(f"💾 Persisted FAISS index to disk: {self.save_directory}")
            print(f"   📁 Index File     : {index_file.name} ({faiss_size_kb:.2f} KB)")
            print(f"   📁 Metadata File  : {pkl_file.name} ({pkl_size_kb:.2f} KB)")
            print(f"   🎯 Vectors Stored : {total_vectors} | Dimension: {dimension}")
            print("=" * 80 + "\n")

        return vector_store

    def load(self) -> FAISS:
        if not self.exists():
            raise FileNotFoundError(f"FAISS index not found in: {self.save_directory}")
        return FAISS.load_local(
            str(self.save_directory),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def exists(self) -> bool:
        index_file = self.save_directory / "index.faiss"
        pkl_file = self.save_directory / "index.pkl"
        return index_file.exists() and pkl_file.exists()
