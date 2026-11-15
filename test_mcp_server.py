"""Tests for the MCP server (mock mode: offline, no API key, no charges)."""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Configure before importing the server: offline embeddings, throwaway index.
os.environ["ENABLE_MOCK_MODE"] = "true"
_TMP = tempfile.mkdtemp(prefix="papermind-mcp-test-")
os.environ["PAPERMIND_CHROMA_DIR"] = str(Path(_TMP) / "chroma")

import mcp_server  # noqa: E402
import rag  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402


def expect_error(fn, *args, **kwargs) -> str:
    try:
        fn(*args, **kwargs)
    except ToolError as exc:
        return str(exc)
    raise AssertionError(f"{fn.__name__} should have raised ToolError")


def test_index_list_and_stats():
    """Indexing returns a paper_id that list_papers and paper_stats agree on."""
    info = mcp_server.index_paper("paper.pdf")
    assert info.paper_id.startswith("paper-")
    assert info.file_name == "paper.pdf"
    assert info.chunks > 0

    again = mcp_server.index_paper("paper.pdf")
    assert again.paper_id == info.paper_id
    assert again.chunks == info.chunks  # reused, not embedded twice

    listed = mcp_server.list_papers()
    assert [p.paper_id for p in listed] == [info.paper_id]
    assert mcp_server.paper_stats(info.paper_id) == info


def test_search_returns_cited_excerpts():
    """Search honours k per call and returns the right passage with its page."""
    paper_id = mcp_server.index_paper("paper.pdf").paper_id

    hits = mcp_server.search_paper(paper_id, "What are the limitations?", k=2)
    assert 1 <= len(hits) <= 2
    assert all(h.page >= 1 and h.text.strip() for h in hits)
    assert any("quadratically" in h.text.lower() for h in hits)
    assert any(h.page == 2 for h in hits)

    assert len(mcp_server.search_paper(paper_id, "attention", k=1)) == 1


def test_input_validation():
    """Bad arguments produce clear errors instead of touching files or the index."""
    paper_id = mcp_server.index_paper("paper.pdf").paper_id

    assert "inside" in expect_error(mcp_server.index_paper, "../../etc/passwd")
    assert "inside" in expect_error(mcp_server.index_paper, str(Path(_TMP) / "x.pdf"))
    assert "chunk_overlap" in expect_error(mcp_server.index_paper, "paper.pdf", 500, 500)
    assert "Unknown paper_id" in expect_error(mcp_server.search_paper, "paper-nope" + mcp_server.model_suffix(), "x")
    assert "different embedding model" in expect_error(mcp_server.search_paper, "paper-abc-openai", "x")
    assert "k must be" in expect_error(mcp_server.search_paper, paper_id, "x", 0)
    assert "empty" in expect_error(mcp_server.search_paper, paper_id, "   ")


def test_extra_pdf_folder():
    """PAPERMIND_PDF_DIRS lets index_paper read from another folder."""
    extra = Path(_TMP) / "papers"
    extra.mkdir(exist_ok=True)
    copy = extra / "copy.pdf"
    shutil.copy(rag.SAMPLE_PDF, copy)

    notes = extra / "notes.txt"
    notes.write_text("not a paper")

    assert "inside" in expect_error(mcp_server.index_paper, str(copy))
    os.environ["PAPERMIND_PDF_DIRS"] = str(extra)
    try:
        assert "not a PDF" in expect_error(mcp_server.index_paper, str(notes))
        assert mcp_server.index_paper(str(copy)).chunks > 0
        assert mcp_server.index_paper("copy.pdf").chunks > 0
    finally:
        del os.environ["PAPERMIND_PDF_DIRS"]


def test_stdio_end_to_end():
    """Start the server as a subprocess and call it like a real MCP client would."""

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(rag.BASE_DIR / "mcp_server.py")],
            env={**os.environ},
            cwd=str(rag.BASE_DIR),
        )
        async with Client(params) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == {"list_papers", "index_paper", "search_paper", "paper_stats"}

            indexed = await client.call_tool("index_paper", {"pdf_path": "paper.pdf"})
            assert not indexed.is_error
            paper_id = indexed.structured_content["paper_id"]

            found = await client.call_tool(
                "search_paper", {"paper_id": paper_id, "query": "BLEU results", "k": 3}
            )
            assert not found.is_error
            assert "BLEU" in str(found.structured_content)

            # Errors must reach the model as readable text, not a generic failure.
            bad = await client.call_tool("search_paper", {"paper_id": "nope", "query": "x"})
            assert bad.is_error
            assert "list_papers" in str(bad.content)

    asyncio.run(run())


if __name__ == "__main__":
    test_index_list_and_stats()
    test_search_returns_cited_excerpts()
    test_input_validation()
    test_extra_pdf_folder()
    test_stdio_end_to_end()
    shutil.rmtree(_TMP, ignore_errors=True)
    print("All MCP server tests passed successfully!")
