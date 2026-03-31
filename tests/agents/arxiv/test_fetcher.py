"""Unit tests for mara.agents.arxiv.fetcher."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.arxiv.fetcher import (
    LATEX,
    chunks_from_pdf,
    compile_latex_to_pdf,
    fetch_source_tarball,
)
from mara.agents.utils.pdf import extract_pdf_text


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


def _make_mock_client(
    status: int = 200,
    content: bytes = b"",
    content_type: str = "application/x-tar",
) -> AsyncMock:
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.headers = {"content-type": content_type}
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    return client


def _fake_pdflatex(cmd: list[str], **kwargs: object) -> MagicMock:
    """subprocess.run side_effect that creates a fake PDF next to the input .tex."""
    main_tex = Path(cmd[-1])
    main_tex.with_suffix(".pdf").write_bytes(b"%PDF-1.4 fake compiled")
    return MagicMock(returncode=0)


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
# compile_latex_to_pdf
# ---------------------------------------------------------------------------


class TestCompileLatexToPdf:
    def test_success_with_main_tex(self):
        tarball = _make_tarball({"main.tex": b"\\documentclass{article}\\begin{document}Hello\\end{document}"})
        with patch("subprocess.run", side_effect=_fake_pdflatex):
            result = compile_latex_to_pdf(tarball)
        assert result == b"%PDF-1.4 fake compiled"

    def test_finds_documentclass_when_no_main_tex(self):
        tarball = _make_tarball({"paper.tex": b"\\documentclass{article}\\begin{document}Body\\end{document}"})
        with patch("subprocess.run", side_effect=_fake_pdflatex):
            result = compile_latex_to_pdf(tarball)
        assert result == b"%PDF-1.4 fake compiled"

    def test_raises_when_pdflatex_produces_no_pdf(self):
        tarball = _make_tarball({"main.tex": b"\\documentclass{article}\\begin{document}\\end{document}"})
        # subprocess.run does nothing — no PDF is created
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            with pytest.raises(RuntimeError, match="pdflatex did not produce"):
                compile_latex_to_pdf(tarball)

    def test_raises_on_corrupt_tarball(self):
        with pytest.raises(Exception):
            compile_latex_to_pdf(b"not a tarball")

    def test_pdflatex_called_with_nonstopmode(self):
        tarball = _make_tarball({"main.tex": b"\\documentclass{article}\\begin{document}\\end{document}"})
        with patch("subprocess.run", side_effect=_fake_pdflatex) as mock_run:
            compile_latex_to_pdf(tarball)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "pdflatex"
        assert "-interaction=nonstopmode" in cmd


# ---------------------------------------------------------------------------
# extract_pdf_text (via utils.pdf)
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
# chunks_from_pdf
# ---------------------------------------------------------------------------


class TestChunksFromPdf:
    def test_success_returns_single_latex_chunk(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Compiled PDF content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            chunks = chunks_from_pdf(
                b"fake", "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00"
            )

        assert len(chunks) == 1
        assert chunks[0].source_type == LATEX
        assert "Compiled PDF content." in chunks[0].text

    def test_extraction_failure_returns_empty(self):
        with patch("pypdf.PdfReader", side_effect=Exception("bad")):
            chunks = chunks_from_pdf(
                b"bad", "https://arxiv.org/abs/123v1", "q", "2024-01-01T00:00:00+00:00"
            )
        assert chunks == []

    def test_url_and_sub_query_propagated(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "text"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            chunks = chunks_from_pdf(
                b"fake", "https://arxiv.org/abs/123v1", "my query", "2024-01-01T00:00:00+00:00"
            )
        assert chunks[0].url == "https://arxiv.org/abs/123v1"
        assert chunks[0].sub_query == "my query"
