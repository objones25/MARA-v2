"""Tests for mara.agents.pwc.agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mara.agents.pwc.agent import (
    PAPER_ABSTRACT,
    PapersWithCodeAgent,
    _parse_paper_results,
)
from mara.agents.registry import AgentConfig
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


def _make_agent(max_results: int = 20) -> PapersWithCodeAgent:
    cfg = _config()
    agent_config = AgentConfig(max_results=max_results)
    return PapersWithCodeAgent(config=cfg, agent_config=agent_config)


def _http_response(data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_paper_data(*items) -> list:
    """Build a minimal HuggingFace Papers API response list.

    Each item is a dict with keys: id, title, summary.
    """
    return [
        {
            "id": item.get("id", "paper-slug"),
            "title": item.get("title", "A Great Paper"),
            "summary": item.get("summary", "This is the summary."),
        }
        for item in items
    ]


def _mock_client_for(responses: list[MagicMock]) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(side_effect=responses)
    return mock_client


# ---------------------------------------------------------------------------
# _parse_paper_results
# ---------------------------------------------------------------------------


class TestParsePaperResults:
    def test_happy_path_returns_url_and_text(self):
        data = _make_paper_data(
            {"id": "gpt-4", "title": "GPT-4 Technical Report", "summary": "We report GPT-4."}
        )
        result = _parse_paper_results(data)
        assert len(result) == 1
        assert result[0]["url"] == "https://huggingface.co/papers/gpt-4"
        assert "GPT-4 Technical Report" in result[0]["text"]
        assert "We report GPT-4." in result[0]["text"]

    def test_missing_paper_id_skipped(self):
        data = [{"id": "", "title": "T", "summary": "S"}]
        assert _parse_paper_results(data) == []

    def test_whitespace_paper_id_skipped(self):
        data = [{"id": "   ", "title": "T", "summary": "S"}]
        assert _parse_paper_results(data) == []

    def test_missing_summary_skipped(self):
        data = [{"id": "p1", "title": "T", "summary": ""}]
        assert _parse_paper_results(data) == []

    def test_whitespace_summary_skipped(self):
        data = [{"id": "p1", "title": "T", "summary": "   "}]
        assert _parse_paper_results(data) == []

    def test_summary_stripped(self):
        data = _make_paper_data({"id": "p1", "title": "T", "summary": "  summary text  "})
        result = _parse_paper_results(data)
        assert result[0]["text"].endswith("summary text")

    def test_title_prepended_to_summary(self):
        data = _make_paper_data({"id": "p1", "title": "My Title", "summary": "My summary."})
        result = _parse_paper_results(data)
        text = result[0]["text"]
        assert text.startswith("My Title")
        assert "My summary." in text

    def test_missing_title_returns_just_summary(self):
        data = [{"id": "p1", "title": "", "summary": "Just the summary."}]
        result = _parse_paper_results(data)
        assert result[0]["text"] == "Just the summary."

    def test_multiple_papers(self):
        data = _make_paper_data(
            {"id": "p1", "title": "T1", "summary": "S1"},
            {"id": "p2", "title": "T2", "summary": "S2"},
        )
        assert len(_parse_paper_results(data)) == 2

    def test_empty_list(self):
        assert _parse_paper_results([]) == []

    def test_null_id_skipped(self):
        data = [{"id": None, "title": "T", "summary": "S"}]
        assert _parse_paper_results(data) == []

    def test_null_summary_skipped(self):
        data = [{"id": "p1", "title": "T", "summary": None}]
        assert _parse_paper_results(data) == []

    @pytest.mark.parametrize(
        "paper_id, expected_url",
        [
            ("my-paper", "https://huggingface.co/papers/my-paper"),
            ("paper-123", "https://huggingface.co/papers/paper-123"),
        ],
        ids=["slug", "slug-with-number"],
    )
    def test_url_construction(self, paper_id, expected_url):
        data = _make_paper_data({"id": paper_id, "title": "T", "summary": "S"})
        result = _parse_paper_results(data)
        assert result[0]["url"] == expected_url


# ---------------------------------------------------------------------------
# PapersWithCodeAgent._chunk
# ---------------------------------------------------------------------------


class TestPapersWithCodeAgentChunk:
    def test_chunk_returns_raw_unchanged(self):
        ag = _make_agent()
        sq = SubQuery(query="test")
        chunks = [
            RawChunk(
                url="https://huggingface.co/papers/p1",
                text="some text",
                retrieved_at="2024-01-01T00:00:00+00:00",
                source_type=PAPER_ABSTRACT,
                sub_query=sq.query,
            )
        ]
        assert ag._chunk(chunks) is chunks


# ---------------------------------------------------------------------------
# PapersWithCodeAgent._search
# ---------------------------------------------------------------------------


class TestPapersWithCodeAgentSearch:
    async def test_happy_path_returns_raw_chunks(self, mocker):
        data = _make_paper_data({"id": "p1", "title": "T", "summary": "S"})
        mock_client = _mock_client_for([_http_response(data)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        ag = _make_agent()
        result = await ag._search(SubQuery(query="image classification"))

        assert len(result) == 1
        assert all(isinstance(c, RawChunk) for c in result)

    def test_paper_chunks_have_correct_source_type(self):
        chunk = RawChunk(
            url="https://huggingface.co/papers/p1",
            text="text",
            retrieved_at="2024-01-01T00:00:00+00:00",
            source_type=PAPER_ABSTRACT,
            sub_query="test",
        )
        assert chunk.source_type == PAPER_ABSTRACT

    async def test_sub_query_field_set_on_all_chunks(self, mocker):
        data = _make_paper_data(
            {"id": "p1", "title": "T1", "summary": "S1"},
            {"id": "p2", "title": "T2", "summary": "S2"},
        )
        mock_client = _mock_client_for([_http_response(data)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        query = "image segmentation"
        result = await _make_agent()._search(SubQuery(query=query))
        assert all(c.sub_query == query for c in result)

    async def test_single_endpoint_queried(self, mocker):
        data = _make_paper_data()
        mock_client = _mock_client_for([_http_response(data)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        assert mock_client.get.call_count == 1

    async def test_limit_param_matches_config(self, mocker):
        data = _make_paper_data()
        mock_client = _mock_client_for([_http_response(data)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent(max_results=5)._search(SubQuery(query="test"))

        _, kwargs = mock_client.get.call_args
        assert kwargs.get("params", {}).get("limit") == 5

    async def test_http_error_propagates(self, mocker):
        mock_client = _mock_client_for([_http_response({}, status_code=429)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_empty_response_returns_empty_list(self, mocker):
        mock_client = _mock_client_for([_http_response([])])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_query_param_sent(self, mocker):
        data = _make_paper_data()
        mock_client = _mock_client_for([_http_response(data)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="object detection"))

        _, kwargs = mock_client.get.call_args
        assert kwargs.get("params", {}).get("q") == "object detection"

    async def test_hf_api_url_used(self, mocker):
        data = _make_paper_data()
        mock_client = _mock_client_for([_http_response(data)])
        mocker.patch("mara.agents.pwc.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        url_called, _ = mock_client.get.call_args
        assert "huggingface.co/api/papers" in url_called[0]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPapersWithCodeRegistration:
    def test_pwc_registered(self):
        import mara.agents.registry as reg_mod

        assert "pwc" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["pwc"].cls is PapersWithCodeAgent

    def test_default_max_results(self):
        import mara.agents.registry as reg_mod

        config = reg_mod._REGISTRY["pwc"].config
        assert config.max_results == 20

    def test_no_api_key_required(self):
        import mara.agents.registry as reg_mod

        config = reg_mod._REGISTRY["pwc"].config
        assert config.api_key == ""
