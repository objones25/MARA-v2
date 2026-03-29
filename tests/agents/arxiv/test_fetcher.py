"""Unit tests for mara.agents.arxiv.fetcher."""

from __future__ import annotations

import io
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.arxiv.fetcher import (
    ABSTRACT_ONLY,
    LATEX,
    PDF_FROM_TARBALL,
    PDF_RENDERED,
    chunks_from_abstract,
    chunks_from_latex,
    chunks_from_pdf,
    extract_from_tarball,
    extract_pdf_text,
    fetch_source_tarball,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tarball(files: dict[str, bytes]) -> bytes:
    """Build an in-memory .tar.gz from a {name: content} mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_mock_client(status: int = 200, content: bytes = b"", content_type: str = "application/x-tar") -> AsyncMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.headers = {"content-type": content_type}
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    return client


# ---------------------------------------------------------------------------
# fetch_source_tarball
# ---------------------------------------------------------------------------


class TestFetchSourceTarball:
    async def test_success_returns_bytes(self):
        tarball = _make_tarball({"main.tex": b"hello"})
        client = _make_mock_client(content=tarball)
        result = await fetch_source_tarball(client, "2301.07041v2")
        assert result == tarball

    async def test_non_200_returns_none(self):
        client = _make_mock_client(status=404)
        result = await fetch_source_tarball(client, "2301.07041v2")
        assert result is None

    async def test_pdf_content_type_returns_none(self):
        client = _make_mock_client(status=200, content_type="application/pdf")
        result = await fetch_source_tarball(client, "2301.07041v2")
        assert result is None

    async def test_network_exception_returns_none(self):
        import httpx

        client = AsyncMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        result = await fetch_source_tarball(client, "2301.07041v2")
        assert result is None

    async def test_correct_url_used(self):
        client = _make_mock_client()
        await fetch_source_tarball(client, "2301.07041v2")
        call_url = client.get.call_args[0][0]
        assert call_url == "https://arxiv.org/src/2301.07041v2"


# ---------------------------------------------------------------------------
# extract_from_tarball
# ---------------------------------------------------------------------------


class TestExtractFromTarball:
    def test_tex_files_extracted(self):
        tarball = _make_tarball({"main.tex": b"\\section{Intro}\nHello."})
        tex_contents, pdf_bytes = extract_from_tarball(tarball)
        assert len(tex_contents) == 1
        assert "Hello." in tex_contents[0]
        assert pdf_bytes is None

    def test_pdf_file_extracted(self):
        fake_pdf = b"%PDF-1.4 fake pdf bytes"
        tarball = _make_tarball({"paper.pdf": fake_pdf})
        tex_contents, pdf_bytes = extract_from_tarball(tarball)
        assert tex_contents == []
        assert pdf_bytes == fake_pdf

    def test_both_tex_and_pdf(self):
        tarball = _make_tarball(
            {"main.tex": b"\\section{A}\nBody.", "paper.pdf": b"%PDF-1.4"}
        )
        tex_contents, pdf_bytes = extract_from_tarball(tarball)
        assert len(tex_contents) == 1
        assert pdf_bytes == b"%PDF-1.4"

    def test_only_first_pdf_extracted(self):
        tarball = _make_tarball(
            {"a.pdf": b"%PDF-1.4 first", "b.pdf": b"%PDF-1.4 second"}
        )
        _, pdf_bytes = extract_from_tarball(tarball)
        assert pdf_bytes == b"%PDF-1.4 first"

    def test_neither_tex_nor_pdf(self):
        tarball = _make_tarball({"readme.txt": b"hello"})
        tex_contents, pdf_bytes = extract_from_tarball(tarball)
        assert tex_contents == []
        assert pdf_bytes is None

    def test_corrupt_tarball_returns_empty(self):
        tex_contents, pdf_bytes = extract_from_tarball(b"not a tarball")
        assert tex_contents == []
        assert pdf_bytes is None

    def test_non_file_member_skipped(self):
        # Create a tarball with a directory entry
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            dir_info = tarfile.TarInfo(name="subdir")
            dir_info.type = tarfile.DIRTYPE
            tar.addfile(dir_info)
            tex_content = b"\\section{A}\nBody."
            tex_info = tarfile.TarInfo(name="subdir/main.tex")
            tex_info.size = len(tex_content)
            tar.addfile(tex_info, io.BytesIO(tex_content))
        tarball = buf.getvalue()

        tex_contents, _ = extract_from_tarball(tarball)
        assert len(tex_contents) == 1

    def test_multiple_tex_files_all_extracted(self):
        tarball = _make_tarball(
            {"intro.tex": b"Intro content", "methods.tex": b"Methods content"}
        )
        tex_contents, _ = extract_from_tarball(tarball)
        assert len(tex_contents) == 2


# ---------------------------------------------------------------------------
# extract_pdf_text
# ---------------------------------------------------------------------------


class TestExtractPdfText:
    def test_success(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one text."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = extract_pdf_text(b"fake pdf")

        assert result == "Page one text."

    def test_multiple_pages_joined(self):
        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page 1."
        pages[1].extract_text.return_value = "Page 2."
        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = extract_pdf_text(b"fake pdf")

        assert "Page 1." in result
        assert "Page 2." in result

    def test_empty_pages_return_none(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = extract_pdf_text(b"fake pdf")

        assert result is None

    def test_none_page_text_skipped(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = extract_pdf_text(b"fake pdf")

        assert result is None

    def test_exception_returns_none(self):
        with patch("pypdf.PdfReader", side_effect=Exception("bad pdf")):
            result = extract_pdf_text(b"not a pdf")
        assert result is None


# ---------------------------------------------------------------------------
# chunks_from_latex
# ---------------------------------------------------------------------------


class TestChunksFromLatex:
    def test_produces_one_chunk_per_nonempty_section(self):
        tex = r"\section{Intro}" + "\nHello world.\n" + r"\section{Methods}" + "\nTechniques."
        chunks = chunks_from_latex([tex], "https://arxiv.org/abs/123v1", "query", "2024-01-01T00:00:00+00:00")
        assert len(chunks) == 2
        assert all(c.source_type == LATEX for c in chunks)
        assert all(c.url == "https://arxiv.org/abs/123v1" for c in chunks)

    def test_title_prepended_to_body(self):
        tex = r"\section{My Title}" + "\nContent here."
        chunks = chunks_from_latex([tex], "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00")
        assert len(chunks) == 1
        assert "My Title" in chunks[0].text
        assert "Content here." in chunks[0].text

    def test_empty_title_not_prepended(self):
        # No section commands → single chunk with empty title
        tex = "Just plain text with no sections."
        chunks = chunks_from_latex([tex], "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00")
        assert len(chunks) == 1
        assert chunks[0].text == "Just plain text with no sections."

    def test_empty_section_body_after_clean_skipped(self):
        # Section body contains only LaTeX commands that clean to empty
        tex = r"\section{Empty}" + "\n\\label{sec:empty}\n"
        chunks = chunks_from_latex([tex], "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00")
        # Either empty (pylatexenc strips the label) or 1 chunk — must not crash
        for c in chunks:
            assert c.text.strip()  # all chunks have non-empty text

    def test_multiple_tex_files(self):
        tex1 = r"\section{A}" + "\nText A."
        tex2 = r"\section{B}" + "\nText B."
        chunks = chunks_from_latex([tex1, tex2], "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00")
        assert len(chunks) == 2

    def test_sub_query_set_correctly(self):
        tex = r"\section{S}" + "\nBody."
        chunks = chunks_from_latex([tex], "https://arxiv.org/abs/123v1", "my query", "2024-01-01T00:00:00+00:00")
        assert chunks[0].sub_query == "my query"


# ---------------------------------------------------------------------------
# chunks_from_pdf
# ---------------------------------------------------------------------------


class TestChunksFromPdf:
    def test_success_returns_single_chunk(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content here."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            chunks = chunks_from_pdf(
                b"fake", "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00", PDF_FROM_TARBALL
            )

        assert len(chunks) == 1
        assert chunks[0].source_type == PDF_FROM_TARBALL
        assert "PDF content here." in chunks[0].text

    def test_extraction_failure_returns_empty(self):
        with patch("pypdf.PdfReader", side_effect=Exception("bad")):
            chunks = chunks_from_pdf(
                b"bad", "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00", PDF_RENDERED
            )
        assert chunks == []

    def test_source_type_propagated(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "text"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            chunks = chunks_from_pdf(
                b"fake", "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00", PDF_RENDERED
            )
        assert chunks[0].source_type == PDF_RENDERED


# ---------------------------------------------------------------------------
# chunks_from_abstract
# ---------------------------------------------------------------------------


class TestChunksFromAbstract:
    def test_returns_single_abstract_chunk(self):
        chunks = chunks_from_abstract(
            "This paper investigates X.",
            "https://arxiv.org/abs/123v1",
            "query",
            "2024-01-01T00:00:00+00:00",
        )
        assert len(chunks) == 1
        assert chunks[0].source_type == ABSTRACT_ONLY
        assert chunks[0].text == "This paper investigates X."
        assert chunks[0].url == "https://arxiv.org/abs/123v1"
