"""Tests for mara.agents.biorxiv.agent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mara.agents.biorxiv.agent import (
    BIORXIV,
    MEDRXIV,
    BioRxivAgent,
    _parse_biorxiv_response,
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


def _make_agent(max_results: int = 20) -> BioRxivAgent:
    cfg = _config()
    agent_config = AgentConfig(max_results=max_results)
    return BioRxivAgent(config=cfg, agent_config=agent_config)


def _http_response(data: dict, status_code: int = 200) -> MagicMock:
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


def _make_biorxiv_response(*items, server: str = "biorxiv") -> dict:
    """Build a minimal bioRxiv API response.

    Each item is a dict with keys: doi, title, abstract, server (optional).
    """
    collection = []
    for item in items:
        collection.append(
            {
                "doi": item.get("doi", "10.1101/2024.01.01.123456"),
                "title": item.get("title", "A Great Preprint"),
                "abstract": item.get("abstract", "This is the abstract."),
                "date": item.get("date", "2024-01-01"),
                "category": item.get("category", "bioinformatics"),
                "server": item.get("server", server),
            }
        )
    return {
        "messages": [
            {"status": "ok", "total": len(collection), "count": len(collection), "cursor": "0"}
        ],
        "collection": collection,
    }


def _mock_client_for(responses: list[MagicMock]) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(side_effect=responses)
    return mock_client


# ---------------------------------------------------------------------------
# _parse_biorxiv_response
# ---------------------------------------------------------------------------


class TestParseBiorxivResponse:
    def test_happy_path_returns_url_and_text(self):
        data = _make_biorxiv_response(
            {
                "doi": "10.1101/2024.01.01.123456",
                "title": "Deep Learning for Protein Folding",
                "abstract": "We present a new approach.",
                "server": "biorxiv",
            }
        )
        result = _parse_biorxiv_response(data)
        assert len(result) == 1
        assert "biorxiv.org" in result[0]["url"]
        assert "10.1101/2024.01.01.123456" in result[0]["url"]
        assert "Deep Learning for Protein Folding" in result[0]["text"]
        assert "We present a new approach." in result[0]["text"]

    def test_missing_doi_skipped(self):
        data = {
            "collection": [
                {"doi": "", "title": "T", "abstract": "A", "server": "biorxiv"}
            ]
        }
        assert _parse_biorxiv_response(data) == []

    def test_whitespace_doi_skipped(self):
        data = {
            "collection": [
                {"doi": "   ", "title": "T", "abstract": "A", "server": "biorxiv"}
            ]
        }
        assert _parse_biorxiv_response(data) == []

    def test_missing_abstract_skipped(self):
        data = {
            "collection": [
                {"doi": "10.1101/abc", "title": "T", "abstract": "", "server": "biorxiv"}
            ]
        }
        assert _parse_biorxiv_response(data) == []

    def test_whitespace_abstract_skipped(self):
        data = {
            "collection": [
                {"doi": "10.1101/abc", "title": "T", "abstract": "   ", "server": "biorxiv"}
            ]
        }
        assert _parse_biorxiv_response(data) == []

    def test_abstract_stripped(self):
        data = _make_biorxiv_response(
            {"doi": "10.1101/abc", "title": "T", "abstract": "  trimmed  "}
        )
        result = _parse_biorxiv_response(data)
        assert result[0]["text"].endswith("trimmed")

    def test_title_prepended_to_abstract(self):
        data = _make_biorxiv_response(
            {"doi": "10.1101/abc", "title": "My Title", "abstract": "My abstract."}
        )
        result = _parse_biorxiv_response(data)
        text = result[0]["text"]
        assert text.startswith("My Title")
        assert "My abstract." in text

    def test_missing_title_returns_just_abstract(self):
        data = {
            "collection": [
                {"doi": "10.1101/abc", "title": "", "abstract": "Just abstract.", "server": "biorxiv"}
            ]
        }
        result = _parse_biorxiv_response(data)
        assert result[0]["text"] == "Just abstract."

    def test_multiple_papers(self):
        data = _make_biorxiv_response(
            {"doi": "10.1101/a", "title": "T1", "abstract": "A1"},
            {"doi": "10.1101/b", "title": "T2", "abstract": "A2"},
        )
        assert len(_parse_biorxiv_response(data)) == 2

    def test_empty_collection(self):
        assert _parse_biorxiv_response({"collection": []}) == []

    def test_missing_collection_key(self):
        assert _parse_biorxiv_response({}) == []

    @pytest.mark.parametrize(
        "server, expected_domain",
        [
            ("biorxiv", "biorxiv.org"),
            ("medrxiv", "medrxiv.org"),
        ],
        ids=["biorxiv-url", "medrxiv-url"],
    )
    def test_url_domain_by_server(self, server, expected_domain):
        data = _make_biorxiv_response(
            {"doi": "10.1101/test", "title": "T", "abstract": "A", "server": server}
        )
        result = _parse_biorxiv_response(data)
        assert expected_domain in result[0]["url"]

    def test_server_field_in_result(self):
        data = _make_biorxiv_response(
            {"doi": "10.1101/abc", "title": "T", "abstract": "A", "server": "medrxiv"}
        )
        result = _parse_biorxiv_response(data)
        assert result[0]["server"] == "medrxiv"


# ---------------------------------------------------------------------------
# BioRxivAgent._chunk
# ---------------------------------------------------------------------------


class TestBioRxivAgentChunk:
    def test_chunk_returns_raw_unchanged(self):
        ag = _make_agent()
        sq = SubQuery(query="test")
        chunks = [
            RawChunk(
                url="https://www.biorxiv.org/content/10.1101/abc",
                text="some text",
                retrieved_at="2024-01-01T00:00:00+00:00",
                source_type=BIORXIV,
                sub_query=sq.query,
            )
        ]
        assert ag._chunk(chunks) is chunks


# ---------------------------------------------------------------------------
# BioRxivAgent._search
# ---------------------------------------------------------------------------


class TestBioRxivAgentSearch:
    async def test_happy_path_returns_raw_chunks(self, mocker):
        biorxiv_data = _make_biorxiv_response(
            {"doi": "10.1101/a", "title": "CRISPR editing", "abstract": "CRISPR approach.", "server": "biorxiv"}
        )
        medrxiv_data = _make_biorxiv_response(
            {"doi": "10.1101/b", "title": "CRISPR clinical", "abstract": "CRISPR trial.", "server": "medrxiv"}
        )
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="CRISPR"))

        assert len(result) >= 1
        assert all(isinstance(c, RawChunk) for c in result)

    async def test_biorxiv_chunks_have_correct_source_type(self, mocker):
        biorxiv_data = _make_biorxiv_response(
            {"doi": "10.1101/a", "title": "Test paper", "abstract": "Test abstract.", "server": "biorxiv"}
        )
        medrxiv_data = _make_biorxiv_response(server="medrxiv")
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="test"))
        biorxiv_chunks = [c for c in result if c.source_type == BIORXIV]
        assert len(biorxiv_chunks) == 1

    async def test_medrxiv_chunks_have_correct_source_type(self, mocker):
        biorxiv_data = _make_biorxiv_response(server="biorxiv")
        medrxiv_data = _make_biorxiv_response(
            {"doi": "10.1101/b", "title": "Clinical study", "abstract": "Clinical abstract.", "server": "medrxiv"}
        )
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="clinical"))
        medrxiv_chunks = [c for c in result if c.source_type == MEDRXIV]
        assert len(medrxiv_chunks) == 1

    async def test_both_servers_queried(self, mocker):
        mock_client = _mock_client_for(
            [_http_response(_make_biorxiv_response()), _http_response(_make_biorxiv_response())]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        assert mock_client.get.call_count == 2

    async def test_biorxiv_url_queried(self, mocker):
        mock_client = _mock_client_for(
            [_http_response(_make_biorxiv_response()), _http_response(_make_biorxiv_response())]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        urls = [call.args[0] for call in mock_client.get.call_args_list]
        assert any("biorxiv" in url for url in urls)

    async def test_medrxiv_url_queried(self, mocker):
        mock_client = _mock_client_for(
            [_http_response(_make_biorxiv_response()), _http_response(_make_biorxiv_response())]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        urls = [call.args[0] for call in mock_client.get.call_args_list]
        assert any("medrxiv" in url for url in urls)

    async def test_keyword_filtering_keeps_matching(self, mocker):
        biorxiv_data = _make_biorxiv_response(
            {
                "doi": "10.1101/match",
                "title": "RNA sequencing of neurons",
                "abstract": "We perform RNA-seq analysis.",
                "server": "biorxiv",
            },
            {
                "doi": "10.1101/nomatch",
                "title": "Climate change effects",
                "abstract": "This paper studies climate.",
                "server": "biorxiv",
            },
        )
        medrxiv_data = _make_biorxiv_response(server="medrxiv")
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="RNA sequencing"))
        urls = [c.url for c in result]
        assert any("match" in url for url in urls)
        assert not any("nomatch" in url for url in urls)

    async def test_keyword_filtering_excludes_non_matching(self, mocker):
        biorxiv_data = _make_biorxiv_response(
            {
                "doi": "10.1101/irrelevant",
                "title": "Economics study",
                "abstract": "Supply and demand.",
                "server": "biorxiv",
            }
        )
        medrxiv_data = _make_biorxiv_response(server="medrxiv")
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="CRISPR gene editing"))
        assert result == []

    async def test_sub_query_field_set_on_all_chunks(self, mocker):
        biorxiv_data = _make_biorxiv_response(
            {"doi": "10.1101/a", "title": "Protein structure", "abstract": "Protein folding study.", "server": "biorxiv"}
        )
        medrxiv_data = _make_biorxiv_response(
            {"doi": "10.1101/b", "title": "Protein interaction", "abstract": "Protein binding data.", "server": "medrxiv"}
        )
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        query = "protein"
        result = await _make_agent()._search(SubQuery(query=query))
        assert all(c.sub_query == query for c in result)

    async def test_both_empty_returns_empty_list(self, mocker):
        mock_client = _mock_client_for(
            [
                _http_response({"messages": [], "collection": []}),
                _http_response({"messages": [], "collection": []}),
            ]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_http_error_propagates(self, mocker):
        mock_client = _mock_client_for([_http_response({}, status_code=503)])
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_max_results_respected(self, mocker):
        many_items = [
            {"doi": f"10.1101/p{i}", "title": f"test paper {i}", "abstract": f"test abstract {i}", "server": "biorxiv"}
            for i in range(10)
        ]
        biorxiv_data = _make_biorxiv_response(*many_items)
        medrxiv_items = [
            {"doi": f"10.1101/m{i}", "title": f"test med {i}", "abstract": f"test med abstract {i}", "server": "medrxiv"}
            for i in range(10)
        ]
        medrxiv_data = _make_biorxiv_response(*medrxiv_items, server="medrxiv")
        mock_client = _mock_client_for(
            [_http_response(biorxiv_data), _http_response(medrxiv_data)]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        result = await _make_agent(max_results=6)._search(SubQuery(query="test"))
        assert len(result) <= 6

    async def test_date_window_in_url(self, mocker):
        """Date range parameters appear in the request URL."""
        mock_client = _mock_client_for(
            [_http_response(_make_biorxiv_response()), _http_response(_make_biorxiv_response())]
        )
        mocker.patch("mara.agents.biorxiv.agent.httpx.AsyncClient", return_value=mock_client)

        await _make_agent()._search(SubQuery(query="test"))

        # URLs must contain a date (YYYY-MM-DD format) for the date window
        import re
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for call in mock_client.get.call_args_list:
            url = call.args[0]
            assert date_pattern.search(url), f"No date found in URL: {url}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestBioRxivRegistration:
    def test_biorxiv_registered(self):
        import mara.agents.registry as reg_mod

        assert "biorxiv" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["biorxiv"].cls is BioRxivAgent

    def test_default_max_results(self):
        import mara.agents.registry as reg_mod

        config = reg_mod._REGISTRY["biorxiv"].config
        assert config.max_results == 20

    def test_no_api_key_required(self):
        import mara.agents.registry as reg_mod

        config = reg_mod._REGISTRY["biorxiv"].config
        assert config.api_key == ""
