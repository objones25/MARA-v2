"""Tests for mara.agents.core.agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.registry import AgentConfig
from mara.agents.core.agent import (
    ABSTRACT_ONLY,
    FULLTEXT,
    PDF_DOWNLOADED,
    COREAgent,
)
from mara.agents.types import RawChunk, SubQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED = {
    "brave_api_key": "brave-key",
    "hf_token": "hf-token",
    "firecrawl_api_key": "fc-key",
    "core_api_key": "core-key",
    "s2_api_key": "s2-key",
    "ncbi_api_key": "ncbi-key",
}


def _config(**overrides):
    from mara.config import ResearchConfig

    return ResearchConfig(**{**_REQUIRED, **overrides}, _env_file=None)


def _make_agent(**cfg_overrides) -> COREAgent:
    max_results = cfg_overrides.pop("core_max_results", None)
    cfg = _config(**cfg_overrides)
    agent_config = cfg.agent_config_overrides.get("core", AgentConfig())
    if max_results is not None:
        agent_config = AgentConfig(
            api_key=agent_config.api_key,
            max_results=max_results,
            rate_limit_rps=agent_config.rate_limit_rps,
        )
    return COREAgent(config=cfg, agent_config=agent_config)


def _json_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = ""
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _bytes_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_search_response(results: list[dict]) -> dict:
    return {"results": results, "totalHits": len(results)}


def _make_work_item(
    work_id: int = 1,
    title: str = "Test Paper",
    abstract: str | None = None,
    full_text: str | None = None,
    download_url: str | None = None,
) -> dict:
    return {
        "id": work_id,
        "title": title,
        "abstract": abstract,
        "fullText": full_text,
        "downloadUrl": download_url,
    }


def _mock_client(responses: list[MagicMock]) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# COREAgent._chunk
# ---------------------------------------------------------------------------


class TestCOREAgentChunk:
    def test_chunk_small_text_passes_through(self):
        ag = _make_agent(chunk_size=100, chunk_overlap=20)
        chunk = RawChunk(
            url="https://core.ac.uk/display/1",
            text="Short abstract.",
            retrieved_at="2024-01-01T00:00:00+00:00",
            source_type=ABSTRACT_ONLY,
            sub_query="test",
        )
        result = ag._chunk([chunk])
        assert len(result) == 1
        assert result[0].text == "Short abstract."

    def test_chunk_large_fulltext_splits(self):
        ag = _make_agent(chunk_size=100, chunk_overlap=20)
        long_text = "x" * 250
        chunk = RawChunk(
            url="https://core.ac.uk/display/2",
            text=long_text,
            retrieved_at="2024-01-01T00:00:00+00:00",
            source_type=FULLTEXT,
            sub_query="test",
        )
        result = ag._chunk([chunk])
        assert len(result) > 1
        assert all(len(c.text) <= 100 for c in result)
        assert all(c.url == chunk.url for c in result)
        assert all(c.source_type == FULLTEXT for c in result)

    def test_chunk_large_pdf_splits(self):
        ag = _make_agent(chunk_size=100, chunk_overlap=20)
        long_text = "y" * 300
        chunk = RawChunk(
            url="https://core.ac.uk/display/3",
            text=long_text,
            retrieved_at="2024-01-01T00:00:00+00:00",
            source_type=PDF_DOWNLOADED,
            sub_query="test",
        )
        result = ag._chunk([chunk])
        assert len(result) > 1
        assert all(c.source_type == PDF_DOWNLOADED for c in result)


# ---------------------------------------------------------------------------
# COREAgent._search
# ---------------------------------------------------------------------------


class TestCOREAgentSearch:
    async def test_happy_path_fulltext_returns_raw_chunks(self, mocker):
        work = _make_work_item(work_id=42, full_text="This is the full text.")
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert isinstance(result[0], RawChunk)
        assert result[0].source_type == FULLTEXT
        assert result[0].text == "This is the full text."

    async def test_happy_path_pdf_fallback(self, mocker):
        work = _make_work_item(
            work_id=7,
            download_url="http://example.com/paper.pdf",
            abstract="The abstract.",
        )
        search = _json_response(_make_search_response([work]))
        pdf_resp = _bytes_response(b"fake-pdf-bytes")
        client = _mock_client([search, pdf_resp])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)
        mocker.patch(
            "mara.agents.core.agent.extract_pdf_text", return_value="Extracted PDF text"
        )

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert result[0].source_type == PDF_DOWNLOADED
        assert result[0].text == "Extracted PDF text"

    async def test_happy_path_abstract_fallback(self, mocker):
        work = _make_work_item(work_id=9, abstract="Just the abstract.")
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert result[0].source_type == ABSTRACT_ONLY
        assert result[0].text == "Just the abstract."

    async def test_source_type_fulltext(self, mocker):
        work = _make_work_item(full_text="Content here.")
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert all(c.source_type == FULLTEXT for c in result)

    async def test_source_type_pdf_downloaded(self, mocker):
        work = _make_work_item(download_url="http://example.com/paper.pdf")
        search = _json_response(_make_search_response([work]))
        pdf_resp = _bytes_response(b"pdf")
        client = _mock_client([search, pdf_resp])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)
        mocker.patch(
            "mara.agents.core.agent.extract_pdf_text", return_value="pdf text"
        )

        result = await _make_agent()._search(SubQuery(query="test"))
        assert all(c.source_type == PDF_DOWNLOADED for c in result)

    async def test_source_type_abstract_only(self, mocker):
        work = _make_work_item(abstract="Abstract only.")
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert all(c.source_type == ABSTRACT_ONLY for c in result)

    async def test_sub_query_field_set(self, mocker):
        work = _make_work_item(full_text="text")
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        query = "machine learning"
        result = await _make_agent()._search(SubQuery(query=query))
        assert all(c.sub_query == query for c in result)

    async def test_url_contains_paper_id(self, mocker):
        work = _make_work_item(work_id=999, full_text="text")
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result[0].url == "https://core.ac.uk/display/999"

    async def test_empty_results_returns_empty(self, mocker):
        search = _json_response(_make_search_response([]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_api_key_sent_in_header(self, mocker):
        search = _json_response(_make_search_response([]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        await _make_agent(core_api_key="my-secret-key")._search(SubQuery(query="test"))

        _, kwargs = client.get.call_args_list[0]
        assert kwargs.get("headers", {}).get("Authorization") == "Bearer my-secret-key"

    async def test_limit_matches_config(self, mocker):
        search = _json_response(_make_search_response([]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        await _make_agent(core_max_results=7)._search(SubQuery(query="test"))

        _, kwargs = client.get.call_args_list[0]
        assert kwargs.get("params", {}).get("limit") == 7

    async def test_http_error_propagates(self, mocker):
        import httpx

        search = _json_response({}, status_code=500)
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_pdf_download_failure_falls_back_to_abstract(self, mocker):
        work = _make_work_item(
            download_url="http://example.com/paper.pdf", abstract="Fallback abstract."
        )
        search = _json_response(_make_search_response([work]))
        pdf_resp = _bytes_response(b"", status_code=404)
        client = _mock_client([search, pdf_resp])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert result[0].source_type == ABSTRACT_ONLY
        assert result[0].text == "Fallback abstract."

    async def test_pdf_extraction_failure_falls_back_to_abstract(self, mocker):
        work = _make_work_item(
            download_url="http://example.com/paper.pdf", abstract="Fallback abstract."
        )
        search = _json_response(_make_search_response([work]))
        pdf_resp = _bytes_response(b"not-a-pdf")
        client = _mock_client([search, pdf_resp])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.core.agent.extract_pdf_text", return_value=None)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert result[0].source_type == ABSTRACT_ONLY

    async def test_no_content_skips_paper(self, mocker):
        """A work with no fullText, no downloadUrl, and no abstract is skipped."""
        work = _make_work_item()
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_multiple_papers_returns_multiple_chunks(self, mocker):
        works = [
            _make_work_item(work_id=1, full_text="text one"),
            _make_work_item(work_id=2, abstract="abstract two"),
        ]
        search = _json_response(_make_search_response(works))
        client = _mock_client([search])
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert len(result) == 2

    async def test_pdf_download_exception_falls_back_to_abstract(self, mocker):
        """Network exception during PDF download → fall back to abstract."""
        work = _make_work_item(
            download_url="http://example.com/paper.pdf", abstract="Safe abstract."
        )
        search = _json_response(_make_search_response([work]))
        client = _mock_client([search])
        client.get = AsyncMock(
            side_effect=[
                search,
                Exception("connection error"),
            ]
        )
        mocker.patch("mara.agents.core.agent.httpx.AsyncClient", return_value=client)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert result[0].source_type == ABSTRACT_ONLY


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestCORERegistration:
    def test_core_registered(self):
        import mara.agents.registry as reg_mod

        assert "core" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["core"].cls is COREAgent
