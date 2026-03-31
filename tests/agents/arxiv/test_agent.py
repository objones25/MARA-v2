"""Tests for mara.agents.arxiv.agent — ArxivAgent, _parse_feed, _versioned_id_from_url."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mara.agents.arxiv.agent import ArxivAgent, _parse_feed, _versioned_id_from_url
from mara.agents.arxiv.fetcher import LATEX
from mara.agents.registry import AgentConfig
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


def _make_chunk(source_type: str = LATEX) -> RawChunk:
    return RawChunk(
        url="https://arxiv.org/abs/1234v1",
        text="A" * 2000,
        retrieved_at="2024-01-01T00:00:00+00:00",
        source_type=source_type,
        sub_query="test query",
    )


def _mock_discovery_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


def _mock_client(discovery_resp: MagicMock) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=discovery_resp)
    return client


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
    def test_sliding_window_applied_to_latex(self, config):
        agent = ArxivAgent(config, AgentConfig())
        result = agent._chunk([_make_chunk(LATEX)])
        # 2000 chars with chunk_size=1000 and overlap=200 → multiple chunks
        assert len(result) > 1

    def test_empty_input(self, config):
        agent = ArxivAgent(config, AgentConfig())
        assert agent._chunk([]) == []


# ---------------------------------------------------------------------------
# ArxivAgent._search — mocked at the fetcher function boundary
# ---------------------------------------------------------------------------


class TestArxivAgentSearch:
    async def test_success_returns_latex_chunks(self, config, mocker):
        """Tarball available, compile succeeds → LATEX chunks returned."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2", "summary": "Abstract text."}])

        mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=b"fake-tarball"),
        )
        fake_pdf = b"%PDF-1.4 fake"
        mocker.patch("mara.agents.arxiv.agent.compile_latex_to_pdf", return_value=fake_pdf)
        mocker.patch(
            "mara.agents.arxiv.agent.chunks_from_pdf",
            return_value=[
                RawChunk(
                    url="https://arxiv.org/abs/2301.07041v2",
                    text="Paper content.",
                    retrieved_at="2024-01-01T00:00:00+00:00",
                    source_type=LATEX,
                    sub_query="quantum computing",
                )
            ],
        )
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        chunks = await agent._search(SubQuery(query="quantum computing"))

        assert len(chunks) == 1
        assert chunks[0].source_type == LATEX
        assert chunks[0].url == "https://arxiv.org/abs/2301.07041v2"

    async def test_tarball_none_returns_empty(self, config, mocker):
        """fetch_source_tarball returns None → no chunks for that paper."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2"}])

        mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=None),
        )
        compile_mock = mocker.patch("mara.agents.arxiv.agent.compile_latex_to_pdf")
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []
        compile_mock.assert_not_called()

    async def test_compile_raises_returns_empty(self, config, mocker):
        """compile_latex_to_pdf raises → exception swallowed, no chunks."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2"}])

        mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=b"fake-tarball"),
        )
        mocker.patch(
            "mara.agents.arxiv.agent.compile_latex_to_pdf",
            side_effect=RuntimeError("pdflatex did not produce"),
        )
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []

    async def test_chunks_from_pdf_empty_returns_empty(self, config, mocker):
        """chunks_from_pdf returns [] (PDF extracted no text) → no chunks."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "2301.07041v2"}])

        mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=b"fake-tarball"),
        )
        mocker.patch("mara.agents.arxiv.agent.compile_latex_to_pdf", return_value=b"%PDF-1.4")
        mocker.patch("mara.agents.arxiv.agent.chunks_from_pdf", return_value=[])
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []

    async def test_empty_feed_returns_no_chunks(self, config, mocker):
        """Empty Atom feed → no chunks, no paper fetches."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""

        fetch_mock = mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=None),
        )
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []
        fetch_mock.assert_not_called()

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

        agent = ArxivAgent(config, AgentConfig())
        with pytest.raises(httpx.HTTPStatusError):
            await agent._search(SubQuery(query="test"))

    async def test_rate_limit_sleep_between_papers(self, config, mocker):
        """asyncio.sleep called once before each paper fetch."""
        mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([
            {"vid": "0001.0001v1"},
            {"vid": "0002.0002v1"},
        ])

        mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=None),
        )
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        await agent._search(SubQuery(query="test"))

        assert mock_sleep.call_count == 2
        assert all(c.args[0] == pytest.approx(3.0) for c in mock_sleep.call_args_list)

    async def test_multiple_papers_aggregated(self, config, mocker):
        """Chunks from multiple papers are all returned."""
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)
        feed_xml = _feed([{"vid": "0001.0001v1"}, {"vid": "0002.0002v1"}])

        mocker.patch(
            "mara.agents.arxiv.agent.fetch_source_tarball",
            new=AsyncMock(return_value=b"fake-tarball"),
        )
        mocker.patch("mara.agents.arxiv.agent.compile_latex_to_pdf", return_value=b"%PDF-1.4")

        def _make_chunks(pdf_bytes, url, sub_query, retrieved_at):  # noqa: ANN001
            return [
                RawChunk(
                    url=url,
                    text="content",
                    retrieved_at=retrieved_at,
                    source_type=LATEX,
                    sub_query=sub_query,
                )
            ]

        mocker.patch("mara.agents.arxiv.agent.chunks_from_pdf", side_effect=_make_chunks)
        mocker.patch(
            "mara.agents.arxiv.agent.httpx.AsyncClient",
            return_value=_mock_client(_mock_discovery_response(feed_xml)),
        )

        agent = ArxivAgent(config, AgentConfig())
        chunks = await agent._search(SubQuery(query="test"))

        assert len(chunks) == 2
        urls = {c.url for c in chunks}
        assert "https://arxiv.org/abs/0001.0001v1" in urls
        assert "https://arxiv.org/abs/0002.0002v1" in urls


# ---------------------------------------------------------------------------
# Registry / config
# ---------------------------------------------------------------------------


class TestArxivRegistration:
    async def test_agent_registered_as_arxiv(self, config):
        from mara.agents.registry import _REGISTRY

        assert "arxiv" in _REGISTRY
        assert _REGISTRY["arxiv"].cls is ArxivAgent

    def test_max_concurrent_one(self):
        from mara.agents.registry import _REGISTRY

        assert _REGISTRY["arxiv"].config.max_concurrent == 1

    def test_retry_backoff_base(self):
        from mara.agents.registry import _REGISTRY

        assert _REGISTRY["arxiv"].config.retry_backoff_base == pytest.approx(5.0)

    def test_max_sub_queries_one(self):
        from mara.agents.registry import _REGISTRY

        assert _REGISTRY["arxiv"].config.max_sub_queries == 1

    def test_max_retries_five(self):
        from mara.agents.registry import _REGISTRY

        assert _REGISTRY["arxiv"].config.max_retries == 5
