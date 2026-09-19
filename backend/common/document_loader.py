import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Common Document Loader for Hybrid RAG.
    Supports PDF documents, plain text, markdown, and structured JSON.
    Includes architecture-aware semantic parsing for technical specs
    (e.g. Spotify Web App Architecture) with cross-page stitching.
    """

    def __init__(
        self,
        mode: str = "auto",  # 'auto', 'section', or 'page'
        clean_text: bool = True,
        verbose: bool = True,
    ):
        self.mode = mode
        self.clean_text = clean_text
        self.verbose = verbose

    def _clean_content(self, text: str) -> str:
        if not self.clean_text:
            return text
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join([line for line in lines if line])

    def _is_architecture_spec(self, full_text: str) -> bool:
        indicators = [
            "Tech Stack",
            "Microservices",
            "Data Storage",
            "Frontend Structure",
            "Feature Checklist",
        ]
        matches = sum(1 for ind in indicators if ind.lower() in full_text.lower())
        return matches >= 2

    def _parse_architecture_sections(
        self,
        reader: PdfReader,
        file_path: Path,
    ) -> List[Document]:
        """
        Extract semantic architectural components from the PDF, resolving cross-page breaks.
        """
        pages_text: List[tuple[int, str]] = []
        for i, page in enumerate(reader.pages):
            pages_text.append((i + 1, page.extract_text() or ""))

        top_section_re = re.compile(r"^([1-5])\.\s+(.*)$")
        microservice_re = re.compile(r"^([0-9]{1,2})\)\s+(.*)$")
        feature_re = re.compile(r"^([A-M])\.\s+(.*)$")
        route_group_re = re.compile(r"^- (Public|Authenticated app|Creator portal|Admin):?$")
        sub_re = re.compile(
            r"^(Frontend|Backend|Databases & Storage|Infrastructure & DevOps|"
            r"PostgreSQL|Redis|Elasticsearch/OpenSearch|ClickHouse / Data Warehouse|"
            r"Object Storage \(S3-like\) \+ CDN|Message Queue \(Kafka/NATS/RabbitMQ\)|"
            r"Global UI components):?$"
        )

        doc_title = "Spotify-like Web App – Full Architecture & Feature List"
        current_top_section = "1. Tech Stack Overview"
        current_subsection = "Overview"
        current_type = "overview"
        current_page_start = 1
        current_page_end = 1
        current_lines: List[str] = []

        documents: List[Document] = []

        def flush(page_num: int):
            nonlocal current_lines, current_page_start, current_page_end
            content = "\n".join(current_lines).strip()
            if content and not all(c in "-= " for c in content) and content != "Key pages/routes:":
                structured_content = (
                    f"Document Title: {doc_title}\n"
                    f"Section: {current_top_section}\n"
                    f"Component: {current_subsection}\n"
                    f"Component Type: {current_type.replace('_', ' ').title()}\n"
                    f"Pages: {current_page_start}-{current_page_end}\n\n"
                    f"Content:\n{content}"
                )

                metadata = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "doc_title": doc_title,
                    "section": current_top_section,
                    "component": current_subsection,
                    "component_type": current_type,
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                    "total_pages": len(reader.pages),
                    "char_count": len(structured_content),
                }

                documents.append(Document(page_content=structured_content, metadata=metadata))

            current_lines = []
            current_page_start = page_num
            current_page_end = page_num

        for page_num, page_raw in pages_text:
            lines = page_raw.splitlines()
            for raw_line in lines:
                line = raw_line.strip()
                if not line or line.startswith("====") or line.startswith("----"):
                    continue
                if "Spotify-like Web App – Full Architecture & Feature List" in line:
                    doc_title = line
                    continue

                m_top = top_section_re.match(line)
                if m_top:
                    flush(page_num)
                    current_top_section = line
                    current_subsection = line
                    current_type = "section_header"
                    current_page_start = page_num
                    current_page_end = page_num
                    continue

                m_ms = microservice_re.match(line)
                if m_ms:
                    flush(page_num)
                    current_subsection = line
                    current_type = "microservice"
                    current_page_start = page_num
                    current_page_end = page_num
                    continue

                m_feat = feature_re.match(line)
                if m_feat:
                    flush(page_num)
                    current_subsection = line
                    current_type = "feature_module"
                    current_page_start = page_num
                    current_page_end = page_num
                    continue

                m_route = route_group_re.match(line)
                if m_route:
                    flush(page_num)
                    current_subsection = f"Routes: {m_route.group(1)}"
                    current_type = "frontend_routes"
                    current_page_start = page_num
                    current_page_end = page_num
                    continue

                m_sub = sub_re.match(line.rstrip(":"))
                if m_sub:
                    flush(page_num)
                    current_subsection = line
                    current_type = "subsystem"
                    current_page_start = page_num
                    current_page_end = page_num
                    continue

                if "This document summarizes the architecture" in line:
                    flush(page_num)
                    current_top_section = "Summary"
                    current_subsection = "Architecture Summary"
                    current_type = "summary"
                    current_page_start = page_num
                    current_page_end = page_num
                    current_lines.append(line)
                    continue

                current_lines.append(raw_line)
                current_page_end = page_num

        flush(len(reader.pages))
        return documents

    def _parse_pages_standard(self, reader: PdfReader, file_path: Path) -> List[Document]:
        documents: List[Document] = []
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            cleaned = self._clean_content(extracted)
            if not cleaned:
                continue

            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "page": page_idx + 1,
                    "page_start": page_idx + 1,
                    "page_end": page_idx + 1,
                    "total_pages": total_pages,
                    "char_count": len(cleaned),
                },
            )
            documents.append(doc)

        return documents

    def load_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        file_size_kb = path.stat().st_size / 1024.0

        if self.verbose:
            print("\n" + "=" * 80)
            print(f"📄 [COMMON LOADER] Ingesting PDF: {path.name}")
            print(f"   📁 Full Path: {path}")
            print(f"   📊 File Size: {file_size_kb:.2f} KB")
            print("=" * 80)

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        combined_text = "".join([p.extract_text() or "" for p in reader.pages[:3]])
        use_section_mode = (
            self.mode == "section"
            or (self.mode == "auto" and self._is_architecture_spec(combined_text))
        )

        if use_section_mode:
            if self.verbose:
                print(f"🔍 Detection: Architecture Specification detected across {total_pages} pages.")
                print("⚡ Parsing Strategy: Semantic Section Extraction (cross-page stitching enabled)")
                print("-" * 80)

            documents = self._parse_architecture_sections(reader, path)

            if self.verbose:
                counts_by_type: Dict[str, int] = {}
                for idx, doc in enumerate(documents, start=1):
                    ctype = doc.metadata.get("component_type", "unknown")
                    counts_by_type[ctype] = counts_by_type.get(ctype, 0) + 1
                    comp = doc.metadata.get("component", "Unknown")
                    p_range = f"P{doc.metadata.get('page_start')}-P{doc.metadata.get('page_end')}"
                    print(
                        f"  [{idx:02d}/{len(documents):02d}] [{p_range}] [{ctype.upper()[:12]:12s}] "
                        f"{comp[:35]:35s} | {doc.metadata.get('char_count', 0):4d} chars"
                    )

                print("-" * 80)
                print(f"📊 Extracted {len(documents)} architectural components:")
                for ctype, count in counts_by_type.items():
                    print(f"   • {ctype.replace('_', ' ').title():24s}: {count} component(s)")
                print(f"✅ Common Loader Complete: {len(documents)} structured units ready.")
                print("=" * 80 + "\n")

            return documents

        else:
            if self.verbose:
                print(f"🔍 Parsing Strategy: Standard Page-by-Page Extraction ({total_pages} pages)")
                print("-" * 80)

            documents = self._parse_pages_standard(reader, path)

            if self.verbose:
                for idx, doc in enumerate(documents, start=1):
                    p_num = doc.metadata.get("page", idx)
                    print(f"  [Page {p_num:02d}/{total_pages:02d}] {doc.metadata.get('char_count', 0):5d} characters extracted")
                print(f"✅ Common Loader Complete: {len(documents)} page documents from {path.name}")
                print("=" * 80 + "\n")

            return documents

    def load_txt(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Text file not found: {path}")

        if self.verbose:
            print(f"📄 [COMMON LOADER] Loading Text File: {path.name}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        cleaned = self._clean_content(content)
        if not cleaned:
            return []

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": "txt",
                "page": 1,
                "total_pages": 1,
                "char_count": len(cleaned),
            },
        )
        if self.verbose:
            print(f"✅ Loaded text document: {len(cleaned)} characters")
        return [doc]

    def load_json(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")

        if self.verbose:
            print(f"📄 [COMMON LOADER] Loading JSON File: {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: List[Document] = []
        items = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and "schemes" in entry:
                    dept = entry.get("department_name", "")
                    for scheme in entry.get("schemes", []):
                        sc = dict(scheme)
                        if "department_name" not in sc:
                            sc["department_name"] = dept
                        items.append(sc)
                elif isinstance(entry, dict):
                    items.append(entry)
        elif isinstance(data, dict):
            items.append(data)

        for item in items:
            details = item.get("details", {})
            title = item.get("scheme_title") or details.get("Scheme Title/Name") or item.get("title", "Untitled")
            dept = item.get("department_name") or details.get("Concerned Department") or item.get("department", "")
            desc = details.get("Description") or item.get("description", "")
            benefits = details.get("Types of Benefits") or item.get("benefits", "")
            beneficiaries = details.get("Beneficiaries") or item.get("beneficiaries", "")
            how_to_avail = details.get("How To avail") or item.get("how_to_avail", "")

            content = f"""Title: {title}
Department: {dept}
Beneficiaries: {beneficiaries}
Benefits: {benefits}
How to Avail: {how_to_avail}
Description: {desc}""".strip()

            doc = Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": "json",
                    "title": title,
                    "department": dept,
                    "char_count": len(content),
                },
            )
            documents.append(doc)

        if self.verbose:
            print(f"✅ Loaded {len(documents)} structured records from {path.name}")
        return documents

    def load(self, path_or_dir: Union[str, Path]) -> List[Document]:
        """
        Universal entry point: loads a single file or an entire directory.
        """
        path = Path(path_or_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_dir():
            return self.load_directory(path)
        else:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return self.load_pdf(path)
            elif suffix in [".txt", ".md", ".log", ".csv"]:
                return self.load_txt(path)
            elif suffix == ".json":
                return self.load_json(path)
            else:
                raise ValueError(f"Unsupported format: {suffix}")

    def load_directory(self, dir_path: Union[str, Path], recursive: bool = True) -> List[Document]:
        path = Path(dir_path).resolve()
        pattern = "**/*" if recursive else "*"
        all_documents: List[Document] = []
        supported_exts = {".pdf", ".txt", ".md", ".log", ".csv", ".json"}

        if self.verbose:
            print(f"\n📂 [COMMON LOADER] Scanning Directory: {path}")

        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                try:
                    docs = self.load(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping failed file {file_path.name}: {e}")

        if self.verbose:
            print(f"🎯 Total documents loaded from directory: {len(all_documents)}")
        return all_documents
