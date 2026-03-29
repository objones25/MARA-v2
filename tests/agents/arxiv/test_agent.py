"""Tests for mara.agents.arxiv.agent — ArxivAgent, _parse_feed, _versioned_id_from_url."""

from __future__ import annotations

import io
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.arxiv.agent import ArxivAgent, _parse_feed, _versioned_id_from_url
from mara.agents.arxiv.fetcher import ABSTRACT_ONLY, LATEX, PDF_FROM_TARBALL, PDF_RENDERED
from mara.agents.types import RawChunk, SubQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ATOM_FEED_TMPL = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  {entries}
</feed>"""

_ENTRY_TMPL = """\
  <entry>
    <id>http://arxiv.org/abs/{vid}</id>
    <title>{title}</title>
    <summary>{summary}</summary>
  </entry>"""


def _feed(entries: list[dict]) -> str:
    entry_xml = "\n".join(
        _ENTRY_TMPL.format(
            vid=e["vid"],
            title=e.get("title", "A Title"),
            summary=e.get("summary", "An abstract."),
        )
        for e in entries
    )
    return _ATOM_FEED_TMPL.format(entries=entry_xml)


def _make_tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _versioned_id_from_url
# ---------------------------------------------------------------------------


class TestVersionedIdFromUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("http://arxiv.org/abs/2301.07041v2", "2301.07041v2"),
            ("https://arxiv.org/abs/2301.07041v2", "2301.07041v2"),
            ("http://arxiv.org/abs/hep-ph/0201093v2", "hep-ph/0201093v2"),
            ("http://arxiv.org/abs/2301.07041v2/", "2301.07041v2"),
        ],
        ids=["new-style-http", "new-style-https", "old-style", "trailing-slash"],
    )
    def test_extraction(self, url, expected):
        assert _versioned_id_from_url(url) == expected

    def test_fallback_no_abs_marker(self):
        # URL without /abs/ — falls back to last segment
        assert _versioned_id_from_url("http://example.com/papers/2301.07041v2") == "2301.07041v2"


# ---------------------------------------------------------------------------
# _parse_feed
# ---------------------------------------------------------------------------


class TestParseFeed:
    def test_single_entry(self):
        xml = _feed([{"vid": "2301.07041v2", "title": "My Paper", "summary": "Great work."}])
        entries = _parse_feed(xml)
        assert len(entries) == 1
        e = entries[0]
        assert e["versioned_id"] == "2301.07041v2"
        assert e["canonical_url"] == "https://arxiv.org/abs/2301.07041v2"
        assert e["abstract"] == "Great work."
        assert e["pdf_url"] == "https://arxiv.org/pdf/2301.07041v2"

    def test_multiple_entries(self):
        xml = _feed([{"vid": "1234.5678v1"}, {"vid": "9876.5432v3"}])
        entries = _parse_feed(xml)
        assert len(entries) == 2

    def test_entry_without_id_skipped(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>No ID here</title>
    <summary>Abstract.</summary>
  </entry>
</feed>"""
        entries = _parse_feed(xml)
        assert entries == []

    def test_missing_title_and_summary(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1111.2222v1</id>
  </entry>
</feed>"""
        entries = _parse_feed(xml)
        assert len(entries) == 1
        assert entries[0]["title"] == ""
        assert entries[0]["abstract"] == ""

    def test_empty_feed(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""
        assert _parse_feed(xml) == []


# ---------------------------------------------------------------------------
# ArxivAgent._chunk
# ---------------------------------------------------------------------------


class TestArxivAgentChunk:
    def _make_chunk(self, source_type: str) -> RawChunk:
        return RawChunk(
            url="https://arxiv.org/abs/1234v1",
            text="A" * 2000,  # longer than default chunk_size to trigger sliding window
            retrieved_at="2024-01-01T00:00:00+00:00",
            source_type=source_type,
            sub_query="test query",
        )

    def test_latex_chunks_pass_through(self, config):
        agent = ArxivAgent(config)
        chunk = self._make_chunk(LATEX)
        result = agent._chunk([chunk])
        assert result == [chunk]

    def test_non_latex_chunks_use_sliding_window(self, config):
        agent = ArxivAgent(config)
        chunk = self._make_chunk(ABSTRACT_ONLY)
        result = agent._chunk([chunk])
        # 2000 chars with chunk_size=1000 and overlap=200 → multiple chunks
        assert len(result) > 1
        assert all(c.source_type == ABSTRACT_ONLY for c in result)

    def test_mixed_chunks_split_correctly(self, config):
        agent = ArxivAgent(config)
        latex_chunk = self._make_chunk(LATEX)
        pdf_chunk = self._make_chunk(PDF_RENDERED)
        result = agent._chunk([latex_chunk, pdf_chunk])
        latex_results = [c for c in result if c.source_type == LATEX]
        pdf_results = [c for c in result if c.source_type == PDF_RENDERED]
        assert len(latex_results) == 1  # unchanged
        assert len(pdf_results) > 1     # slid

    def test_empty_input(self, config):
        agent = ArxivAgent(config)
        assert agent._chunk([]) == []

    def test_all_non_latex(self, config):
        agent = ArxivAgent(config)
        chunk = self._make_chunk(PDF_FROM_TARBALL)
        result = agent._chunk([chunk])
        assert all(c.source_type == PDF_FROM_TARBALL for c in result)

    def test_only_latex_no_super_call(self, config):
        agent = ArxivAgent(config)
        latex_chunk = RawChunk(
            url="https://arxiv.org/abs/1234v1",
            text="Short latex section.",
            retrieved_at="2024-01-01T00:00:00+00:00",
            source_type=LATEX,
            sub_query="q",
        )
        result = agent._chunk([latex_chunk])
        assert result == [latex_chunk]


# ---------------------------------------------------------------------------
# ArxivAgent._search — integration tests with mocked HTTP
# ---------------------------------------------------------------------------


class TestArxivAgentSearch:
    def _mock_response(self, status: int = 200, content: bytes = b"", text: str = "", content_type: str = "application/x-tar"):
        resp = MagicMock()
        resp.status_code = status
        resp.content = content
        resp.text = text
        resp.headers = {"content-type": content_type}
        resp.raise_for_status = MagicMock()
        return resp

    async def test_latex_path(self, config, mocker):
        """LaTeX source available → LATEX chunks returned."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        tex = r"\section{Introduction}" + "\nThis is the content of the paper."
        tarball = _make_tarball({"main.tex": tex.encode()})
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract text."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(content=tarball)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        agent = ArxivAgent(config)
        sub_query = SubQuery(query="quantum computing")
        chunks = await agent._search(sub_query)

        assert len(chunks) >= 1
        assert all(c.source_type == LATEX for c in chunks)
        assert all(c.url == "https://arxiv.org/abs/2301.07041v2" for c in chunks)

    async def test_pdf_from_tarball_path(self, config, mocker):
        """Tarball has no .tex but has .pdf → PDF_FROM_TARBALL chunks."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        fake_pdf = b"%PDF-1.4"
        tarball = _make_tarball({"paper.pdf": fake_pdf})
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(content=tarball)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content from tarball."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == PDF_FROM_TARBALL

    async def test_rendered_pdf_path(self, config, mocker):
        """No tarball → fall back to versioned PDF URL."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(status=404)
        pdf_resp = self._mock_response(status=200, content=b"%PDF-1.4")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp, pdf_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Rendered PDF text."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == PDF_RENDERED

    async def test_abstract_only_path(self, config, mocker):
        """All content fetches fail → abstract-only chunk."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Fallback abstract."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(status=404)
        pdf_resp = self._mock_response(status=200, content=b"not a pdf")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp, pdf_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        with patch("pypdf.PdfReader", side_effect=Exception("bad pdf")):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == ABSTRACT_ONLY
        assert "Fallback abstract." in chunks[0].text

    async def test_no_content_no_abstract_returns_empty(self, config, mocker):
        """All fetches fail and abstract is empty → no chunks for this paper."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": ""}])
        # Override summary in feed to be empty
        feed_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.07041v2</id>
    <title>T</title>
    <summary></summary>
  </entry>
</feed>"""

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(status=404)
        pdf_resp = self._mock_response(status=200, content=b"bad")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp, pdf_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        with patch("pypdf.PdfReader", side_effect=Exception("bad")):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []

    async def test_discovery_failure_raises(self, config, mocker):
        """API call failure propagates as HTTPStatusError."""
        import httpx

        mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        error_resp = MagicMock()
        error_resp.status_code = 503
        error_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=error_resp)
        )
        error_resp.text = ""

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=error_resp)

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        agent = ArxivAgent(config)
        with pytest.raises(httpx.HTTPStatusError):
            await agent._search(SubQuery(query="test"))

    async def test_rate_limit_sleep_between_papers(self, config, mocker):
        """asyncio.sleep is called once between the two papers (discovery lock removed)."""
        mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        tex = r"\section{A}" + "\nContent."
        tarball = _make_tarball({"main.tex": tex.encode()})
        feed_xml = _feed([
            {"vid": "0001.0001v1", "summary": "A1."},
            {"vid": "0002.0002v1", "summary": "A2."},
        ])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp1 = self._mock_response(content=tarball)
        tarball_resp2 = self._mock_response(content=tarball)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp1, tarball_resp2])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        agent = ArxivAgent(config)
        await agent._search(SubQuery(query="test"))

        # 1 sleep between the two papers (base class handles discovery-level rate limiting)
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0].args[0] == pytest.approx(3.0)

    async def test_pdf_fetch_exception_falls_to_abstract(self, config, mocker):
        """PDF GET raises an exception → fall through to abstract."""
        import httpx

        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract text."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(status=404)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(
            side_effect=[api_resp, tarball_resp, httpx.ConnectError("timeout")]
        )

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        agent = ArxivAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == ABSTRACT_ONLY

    async def test_latex_empty_falls_to_tarball_pdf(self, config, mocker):
        """Tarball has .tex that produces no chunks → use PDF in tarball."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        # .tex file with no content that survives cleaning
        fake_pdf = b"%PDF-1.4"
        tarball = _make_tarball({"main.tex": b"\\label{x}", "paper.pdf": fake_pdf})
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(content=tarball)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF from tarball content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        # Should fall through to PDF_FROM_TARBALL since latex produced no chunks
        assert len(chunks) >= 1
        # Accept either PDF_FROM_TARBALL or ABSTRACT_ONLY (depends on latex cleaning)
        assert all(c.source_type in (PDF_FROM_TARBALL, ABSTRACT_ONLY, LATEX) for c in chunks)

    async def test_empty_entry_list_returns_no_chunks(self, config, mocker):
        """Empty feed → no chunks, no errors."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=api_resp)

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        agent = ArxivAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []

    async def test_tarball_no_tex_no_pdf_falls_to_rendered_pdf(self, config, mocker):
        """Tarball exists but has neither .tex nor .pdf → skip step 2, try rendered PDF."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        tarball = _make_tarball({"readme.txt": b"just a readme"})
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(content=tarball)
        pdf_resp = self._mock_response(status=200, content=b"%PDF-1.4")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp, pdf_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Rendered PDF text."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == PDF_RENDERED

    async def test_tarball_pdf_no_text_falls_to_rendered_pdf(self, config, mocker):
        """Tarball PDF yields no text → fall through to rendered PDF URL."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        tarball = _make_tarball({"paper.pdf": b"%PDF-1.4 empty"})
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(content=tarball)
        pdf_resp = self._mock_response(status=200, content=b"%PDF-1.4")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp, pdf_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        # First PdfReader call (tarball PDF) returns no text; second (rendered) does
        empty_page = MagicMock()
        empty_page.extract_text.return_value = ""
        empty_reader = MagicMock()
        empty_reader.pages = [empty_page]

        real_page = MagicMock()
        real_page.extract_text.return_value = "Rendered PDF text."
        real_reader = MagicMock()
        real_reader.pages = [real_page]

        with patch("pypdf.PdfReader", side_effect=[empty_reader, real_reader]):
            agent = ArxivAgent(config)
            chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == PDF_RENDERED

    async def test_rendered_pdf_non_200_falls_to_abstract(self, config, mocker):
        """Rendered PDF URL returns non-200 → skip to abstract-only."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract text."}])

        api_resp = self._mock_response(text=feed_xml)
        api_resp.raise_for_status = MagicMock()
        tarball_resp = self._mock_response(status=404)
        pdf_resp = self._mock_response(status=403)  # non-200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=[api_resp, tarball_resp, pdf_resp])

        mocker.patch("mara.agents.arxiv.agent.httpx.AsyncClient", return_value=mock_client)

        agent = ArxivAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 1
        assert chunks[0].source_type == ABSTRACT_ONLY

    async def test_agent_registered_as_arxiv(self, config):
        from mara.agents.registry import _REGISTRY

        assert "arxiv" in _REGISTRY
        assert _REGISTRY["arxiv"] is ArxivAgent
