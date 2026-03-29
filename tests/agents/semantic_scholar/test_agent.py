"""Tests for mara.agents.semantic_scholar.agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.semantic_scholar.agent import (
    SNIPPET,
    SemanticScholarAgent,
    _parse_snippet_response,
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


def _make_agent(**cfg_overrides) -> SemanticScholarAgent:
    return SemanticScholarAgent(config=_config(**cfg_overrides))


def _make_http_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_s2_data(*papers: tuple[str, str]) -> dict:
    """Build a minimal S2 snippet search response.

    Each paper is (corpusId, snippet_text). The real API returns one ``snippet``
    object per result item (not a list), identified by ``corpusId``.
    """
    data = []
    for corpus_id, snippet_text in papers:
        data.append(
            {
                "paper": {"corpusId": corpus_id},
                "snippet": {"text": snippet_text},
            }
        )
    return {"data": data}


# ---------------------------------------------------------------------------
# _parse_snippet_response
# ---------------------------------------------------------------------------


class TestParseSnippetResponse:
    def test_happy_path_returns_url_and_text(self):
        data = _make_s2_data(("203952528", "Quantum entanglement is cool."))
        result = _parse_snippet_response(data)
        assert len(result) == 1
        assert result[0]["url"] == "https://www.semanticscholar.org/paper/203952528"
        assert result[0]["text"] == "Quantum entanglement is cool."

    def test_missing_corpus_id_skipped(self):
        data = {"data": [{"paper": {}, "snippet": {"text": "some text"}}]}
        assert _parse_snippet_response(data) == []

    def test_whitespace_only_snippet_skipped(self):
        data = _make_s2_data(("abc123", "   "))
        assert _parse_snippet_response(data) == []

    def test_empty_snippet_text_skipped(self):
        data = _make_s2_data(("abc123", ""))
        assert _parse_snippet_response(data) == []

    def test_multiple_papers(self):
        data = _make_s2_data(("p1", "Text 1."), ("p2", "Text 2."))
        result = _parse_snippet_response(data)
        assert len(result) == 2
        urls = {r["url"] for r in result}
        assert "https://www.semanticscholar.org/paper/p1" in urls
        assert "https://www.semanticscholar.org/paper/p2" in urls

    def test_empty_data_list(self):
        assert _parse_snippet_response({"data": []}) == []

    def test_missing_data_key(self):
        assert _parse_snippet_response({}) == []

    def test_missing_snippet_key_skipped(self):
        data = {"data": [{"paper": {"corpusId": "abc123"}}]}
        assert _parse_snippet_response(data) == []

    @pytest.mark.parametrize(
        "corpus_id, expected_url",
        [
            ("simple", "https://www.semanticscholar.org/paper/simple"),
            ("203952528", "https://www.semanticscholar.org/paper/203952528"),
        ],
        ids=["string-id", "numeric-id"],
    )
    def test_url_construction(self, corpus_id, expected_url):
        data = _make_s2_data((corpus_id, "some text"))
        result = _parse_snippet_response(data)
        assert result[0]["url"] == expected_url

    def test_text_is_stripped(self):
        data = _make_s2_data(("abc", "  leading and trailing  "))
        result = _parse_snippet_response(data)
        assert result[0]["text"] == "leading and trailing"


# ---------------------------------------------------------------------------
# SemanticScholarAgent._chunk
# ---------------------------------------------------------------------------


class TestSemanticScholarAgentChunk:
    def test_chunk_returns_raw_unchanged(self):
        agent = _make_agent()
        sq = SubQuery(query="test")
        chunks = [
            RawChunk(
                url="https://www.semanticscholar.org/paper/x",
                text="Some text here.",
                retrieved_at="2024-01-01T00:00:00+00:00",
                source_type=SNIPPET,
                sub_query=sq.query,
            )
        ]
        assert agent._chunk(chunks) is chunks


# ---------------------------------------------------------------------------
# SemanticScholarAgent._search
# ---------------------------------------------------------------------------


class TestSemanticScholarAgentSearch:
    def _mock_client(self, response: MagicMock) -> AsyncMock:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=response)
        return mock_client

    async def test_happy_path_returns_raw_chunks(self, mocker):
        data = _make_s2_data(("paper1", "Quantum computing advances."))
        resp = _make_http_response(data)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        agent = _make_agent()
        sq = SubQuery(query="quantum computing")
        result = await agent._search(sq)

        assert len(result) == 1
        assert isinstance(result[0], RawChunk)
        assert result[0].url == "https://www.semanticscholar.org/paper/paper1"
        assert result[0].text == "Quantum computing advances."

    async def test_source_type_is_snippet(self, mocker):
        data = _make_s2_data(("paper1", "Some snippet."))
        resp = _make_http_response(data)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result[0].source_type == SNIPPET

    async def test_sub_query_field_set(self, mocker):
        data = _make_s2_data(("paper1", "text"))
        resp = _make_http_response(data)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        query = "machine learning"
        result = await _make_agent()._search(SubQuery(query=query))
        assert result[0].sub_query == query

    async def test_api_key_header_sent(self, mocker):
        data = _make_s2_data(("paper1", "text"))
        resp = _make_http_response(data)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        _, kwargs = mock_client.get.call_args
        assert kwargs.get("headers", {}).get("x-api-key") == "s2-key"

    async def test_limit_param_matches_config(self, mocker):
        data = _make_s2_data(("paper1", "text"))
        resp = _make_http_response(data)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent(s2_max_results=5)._search(SubQuery(query="test"))

        _, kwargs = mock_client.get.call_args
        assert kwargs.get("params", {}).get("limit") == 5

    async def test_http_error_propagates(self, mocker):
        import httpx

        resp = _make_http_response({}, status_code=429)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_empty_data_returns_empty_list(self, mocker):
        resp = _make_http_response({"data": []})
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_multiple_results(self, mocker):
        data = _make_s2_data(
            ("p1", "Snippet A."),
            ("p2", "Snippet B."),
            ("p3", "Snippet C."),
        )
        resp = _make_http_response(data)
        mock_client = self._mock_client(resp)
        mocker.patch("mara.agents.semantic_scholar.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestSemanticScholarRegistration:
    def test_s2_registered(self):
        import mara.agents.registry as reg_mod

        assert "s2" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["s2"] is SemanticScholarAgent
