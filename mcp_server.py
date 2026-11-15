"""MCP server for the AI Research Paper Assistant.

Exposes the RAG pipeline in rag.py as Model Context Protocol tools, so any MCP
client (Claude Desktop, Claude Code, Cursor, your own agents) can index and
search research papers:

    list_papers   - papers that are already indexed
    index_paper   - index a PDF (load -> split -> embed -> Chroma)
    search_paper  - retrieve the most relevant excerpts, with page numbers
    paper_stats   - details of one indexed paper

The server only retrieves. It returns excerpts and page numbers and leaves the
answer to the client's own model, so no chat-model API key is needed here.
Embeddings follow the same mock / OpenAI switch as the rest of the project.

Run it (stdio transport):
    python mcp_server.py

Notes:
- With stdio, stdout carries the protocol. Never print() from this module or
  anything it imports; log to stderr instead.
- Raise ToolError for failures the client's model should read and act on. Any
  other exception reaches the model only as "Error executing tool <name>".
"""

import os
import threading
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

import rag

MAX_K = 20

mcp = MCPServer(
    "PaperMind",
    instructions=(
        "Search research papers (PDFs) that have been indexed into a vector store. "
        "Call list_papers to see what is available, index_paper to add a PDF, and "
        "search_paper to get relevant excerpts. Answer only from the excerpts and "
        "cite page numbers like (p. 2)."
    ),
)


# --- Configuration ----------------------------------------------------------
def chroma_dir() -> Path:
    """Vector store location (override with PAPERMIND_CHROMA_DIR)."""
    return Path(os.getenv("PAPERMIND_CHROMA_DIR") or rag.CHROMA_DIR)


def allowed_pdf_dirs() -> List[Path]:
    """Folders index_paper may read from.

    Always the project's data/ folder, plus any folders listed in
    PAPERMIND_PDF_DIRS (separated by ';' on Windows, ':' elsewhere).
    """
    dirs = [rag.DATA_DIR]
    extra = os.getenv("PAPERMIND_PDF_DIRS", "")
    dirs += [Path(d).expanduser() for d in extra.split(os.pathsep) if d.strip()]
    return [d.resolve() for d in dirs]


def resolve_pdf(pdf_path: str) -> Path:
    """Turn a tool argument into a real PDF path inside an allowed folder."""
    raw = Path(pdf_path).expanduser()
    roots = allowed_pdf_dirs()
    if raw.is_absolute():
        candidates = [raw]
    else:
        # Relative names are looked up in uploads/ first, then each allowed folder.
        candidates = [rag.UPLOADS_DIR / raw] + [root / raw for root in roots]

    for candidate in candidates:
        path = candidate.resolve()
        if not any(path.is_relative_to(root) for root in roots):
            continue
        if path.is_file():
            if path.suffix.lower() != ".pdf":
                raise ToolError(f"{path.name} is not a PDF file.")
            return path

    allowed = ", ".join(str(r) for r in roots)
    raise ToolError(
        f"No PDF found for '{pdf_path}'. Files must be inside: {allowed}. "
        "Add folders with the PAPERMIND_PDF_DIRS environment variable."
    )


# --- Shared state -----------------------------------------------------------
_lock = threading.Lock()
_embeddings: Optional[Embeddings] = None
_client = None
_client_path: Optional[Path] = None


def get_embeddings() -> Embeddings:
    """One embedding model per server process, chosen like the rest of the app."""
    global _embeddings
    with _lock:
        if _embeddings is None:
            if rag.is_mock_mode():
                _embeddings = rag.MockEmbeddings()
            else:
                _embeddings = OpenAIEmbeddings(model=rag.EMBEDDING_MODEL)
        return _embeddings


def get_client():
    """Chroma client for the persistent store (re-created if the folder changes)."""
    global _client, _client_path
    with _lock:
        path = chroma_dir()
        if _client is None or _client_path != path:
            _client = chromadb.PersistentClient(path=str(path))
            _client_path = path
        return _client


def model_suffix() -> str:
    """Collection-name suffix of the embedding model this server is using."""
    model_name = getattr(get_embeddings(), "model", "custom")
    return "-" + rag.embedding_model_tag(model_name)


# --- Results ----------------------------------------------------------------
class PaperInfo(BaseModel):
    """One indexed paper."""

    paper_id: str = Field(description="Identifier to pass to search_paper and paper_stats")
    file_name: str = Field(description="Name of the source PDF")
    pages: Optional[int] = Field(description="Page count of the PDF, when known")
    chunks: int = Field(description="Number of indexed text chunks")
    embedding_model: str = Field(description="Embedding model the index was built with")


class Excerpt(BaseModel):
    """One retrieved passage."""

    page: int = Field(description="1-based page number in the PDF")
    text: str = Field(description="Passage text")


def describe(collection) -> PaperInfo:
    """Build a PaperInfo from a Chroma collection.

    PyPDFLoader stores the source path and page count on every chunk, so one
    stored chunk is enough to recover them.
    """
    sample = collection.get(limit=1, include=["metadatas"])
    meta = (sample.get("metadatas") or [None])[0] or {}
    source = meta.get("source")
    return PaperInfo(
        paper_id=collection.name,
        file_name=Path(source).name if source else "unknown",
        pages=meta.get("total_pages"),
        chunks=collection.count(),
        embedding_model=getattr(get_embeddings(), "model", "custom"),
    )


def get_collection(paper_id: str):
    """Look up an indexed paper, with errors a model can act on."""
    if not paper_id.endswith(model_suffix()):
        raise ToolError(
            f"Paper '{paper_id}' was indexed with a different embedding model than "
            "this server is using. Call list_papers, or index the PDF again."
        )
    try:
        return get_client().get_collection(paper_id)
    except Exception:
        raise ToolError(
            f"Unknown paper_id '{paper_id}'. Call list_papers to see indexed papers."
        ) from None


# --- Tools ------------------------------------------------------------------
@mcp.tool()
def list_papers() -> List[PaperInfo]:
    """List the papers that are indexed and ready to search."""
    suffix = model_suffix()
    papers = []
    for collection in get_client().list_collections():
        if collection.name.startswith("paper-") and collection.name.endswith(suffix):
            if collection.count():
                papers.append(describe(collection))
    return sorted(papers, key=lambda p: p.file_name.lower())


@mcp.tool()
def index_paper(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> PaperInfo:
    """Index a research paper PDF so it can be searched.

    pdf_path is a file name inside the project's data/ or data/uploads/ folder,
    or a full path inside an allowed folder. Indexing the same file again with
    the same settings reuses the stored index. Returns the paper_id to use with
    search_paper.
    """
    if chunk_size < 100:
        raise ToolError("chunk_size must be at least 100 characters.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ToolError("chunk_overlap must be between 0 and chunk_size - 1.")

    path = resolve_pdf(pdf_path)
    try:
        store = rag.build_vectorstore(
            path,
            chunk_size,
            chunk_overlap,
            persist_dir=chroma_dir(),
            embeddings=get_embeddings(),
        )
    except ValueError as exc:  # e.g. a scanned PDF with no extractable text
        raise ToolError(str(exc)) from exc
    return describe(store._collection)


@mcp.tool()
def search_paper(paper_id: str, query: str, k: int = 4) -> List[Excerpt]:
    """Return the k passages of an indexed paper most relevant to the query.

    Use the passages as the only source for an answer and cite their page numbers.
    """
    if not query.strip():
        raise ToolError("query must not be empty.")
    if not 1 <= k <= MAX_K:
        raise ToolError(f"k must be between 1 and {MAX_K}.")

    get_collection(paper_id)
    store = Chroma(
        client=get_client(),
        collection_name=paper_id,
        embedding_function=get_embeddings(),
    )
    docs = store.similarity_search(query, k=k)
    return [
        Excerpt(page=doc.metadata.get("page", 0) + 1, text=doc.page_content)
        for doc in docs
    ]


@mcp.tool()
def paper_stats(paper_id: str) -> PaperInfo:
    """Return file name, page count, chunk count and embedding model of one indexed paper."""
    return describe(get_collection(paper_id))


if __name__ == "__main__":
    mcp.run()  # stdio transport
