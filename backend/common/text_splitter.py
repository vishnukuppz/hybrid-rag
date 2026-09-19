import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class CommonTextSplitter:
    """
    Common Text Splitter for both Vector and Knowledge Graph pipelines.
    Splits documents into semantic chunks while enriching each chunk with
    section metadata, chunk index, character length, and console visibility.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.verbose = verbose
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into chunked Documents with enriched metadata.
        """
        if not documents:
            if self.verbose:
                print("⚠️ [COMMON SPLITTER] No documents provided for text splitting.")
            return []

        if self.verbose:
            print("\n" + "=" * 80)
            print("✂️ [COMMON STEP: TEXT SPLITTING & CHUNKING]")
            print(f"   ⚙️ Config: chunk_size={self.chunk_size} chars | chunk_overlap={self.chunk_overlap} chars")
            print(f"   📄 Input Documents to Split: {len(documents)}")
            print("-" * 80)

        raw_chunks = self._splitter.split_documents(documents)

        enriched_chunks: List[Document] = []
        for idx, chunk in enumerate(raw_chunks, start=1):
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = idx
            metadata["chunk_char_length"] = len(chunk.page_content)
            enriched_chunks.append(
                Document(page_content=chunk.page_content, metadata=metadata)
            )

        if self.verbose:
            sample_count = min(5, len(enriched_chunks))
            print(f"📦 Generated {len(enriched_chunks)} Chunks total. Showing first {sample_count} samples:")
            for i in range(sample_count):
                chk = enriched_chunks[i]
                comp = chk.metadata.get("component") or chk.metadata.get("section") or f"Doc {chk.metadata.get('page', '?')}"
                preview = chk.page_content.replace("\n", " ")[:90]
                print(f"   • Chunk #{i+1:02d} | [{comp[:30]:30s}] ({len(chk.page_content):4d} chars): \"{preview}...\"")

            if len(enriched_chunks) > sample_count:
                print(f"   ... and {len(enriched_chunks) - sample_count} more chunks.")

            print(f"✅ Common Text Splitting Complete: {len(enriched_chunks)} chunks ready for downstream pipelines.")
            print("=" * 80 + "\n")

        return enriched_chunks
