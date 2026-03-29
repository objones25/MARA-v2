"""Tests for mara.agents.citation_graph.agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mara.agents.citation_graph.agent import (
    CITATION,
    CitationGraphAgent,
    _parse_citations_response,
    _parse_paper_search_response,
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


def _make_agent(max_results: int = 20) -> CitationGraphAgent:
    cfg = _config()
    agent_config = AgentConfig(rate_limit_rps=1.0, max_results=max_results, api_key="s2-key")
    return CitationGraphAgent(config=cfg, agent_config=agent_config)


def _make_http_response(data, status_code: int = 200) -> MagicMock:
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


def _mock_client(responses: list) -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.get = AsyncMock(side_effect=responses)
    return mock


def _make_paper_search_data(corpus_id: str | None = "seed123") -> dict:
    if corpus_id is None:
        return {"data": []}
    return {"data": [{"corpusId": corpus_id, "title": "Seed Paper Title"}]}


def _make_citations_data(*citing_papers: tuple) -> dict:
    """Build a citations API response.

    Each arg is a tuple of (corpus_id, title, abstract, year).
    Pass None to omit a field.
    """
    data = []
    for corpus_id, title, abstract, year in citing_papers:
        paper: dict = {}
        if corpus_id is not None:
            paper["corpusId"] = corpus_id
        if title is not None:
            paper["title"] = title
        if abstract is not None:
            paper["abstract"] = abstract
        if year is not None:
            paper["year"] = year
        data.append({"citingPaper": paper})
    return {"data": data}


# ---------------------------------------------------------------------------
# _parse_paper_search_response
# ---------------------------------------------------------------------------


class TestParsePaperSearchResponse:
    def test_returns_corpus_id_of_first_result(self):
        data = _make_paper_search_data("abc123")
        assert _parse_paper_search_response(data) == "abc123"

    def test_empty_data_list_returns_none(self):
        assert _parse_paper_search_response({"data": []}) is None

    def test_missing_data_key_returns_none(self):
        assert _parse_paper_search_response({}) is None

    def test_missing_corpus_id_in_first_item_returns_none(self):
        data = {"data": [{"title": "Some Paper"}]}
        assert _parse_paper_search_response(data) is None

    def test_numeric_corpus_id_converted_to_str(self):
        data = {"data": [{"corpusId": 12345}]}
        result = _parse_paper_search_response(data)
        assert result == "12345"
        assert isinstance(result, str)

    def test_only_first_result_is_used(self):
        data = {"data": [{"corpusId": "first"}, {"corpusId": "second"}]}
        assert _parse_paper_search_response(data) == "first"


# ---------------------------------------------------------------------------
# _parse_citations_response
# ---------------------------------------------------------------------------


class TestParseCitationsResponse:
    def test_happy_path_all_fields(self):
        data = _make_citations_data(("cid1", "My Title", "My abstract.", 2023))
        result = _parse_citations_response(data)
        assert len(result) == 1
        assert result[0]["url"] == "https://www.semanticscholar.org/paper/cid1"
        assert "2023: My Title" in result[0]["text"]
        assert "My abstract." in result[0]["text"]

    def test_missing_corpus_id_skipped(self):
        data = {"data": [{"citingPaper": {"title": "No ID Paper", "abstract": "Some abstract."}}]}
        assert _parse_citations_response(data) == []

    def test_empty_data_list_returns_empty(self):
        assert _parse_citations_response({"data": []}) == []

    def test_missing_data_key_returns_empty(self):
        assert _parse_citations_response({}) == []

    def test_title_only_no_abstract_no_year(self):
        data = _make_citations_data(("cid1", "Just A Title", None, None))
        result = _parse_citations_response(data)
        assert len(result) == 1
        assert result[0]["text"] == "Just A Title"

    def test_abstract_only_no_title_no_year(self):
        data = _make_citations_data(("cid1", None, "Abstract only.", None))
        result = _parse_citations_response(data)
        assert len(result) == 1
        assert result[0]["text"] == "Abstract only."

    def test_year_prefixes_title(self):
        data = _make_citations_data(("cid1", "My Paper", None, 2021))
        result = _parse_citations_response(data)
        assert result[0]["text"] == "2021: My Paper"

    def test_no_title_no_abstract_skipped(self):
        data = {"data": [{"citingPaper": {"corpusId": "cid1", "year": 2023}}]}
        assert _parse_citations_response(data) == []

    def test_whitespace_title_and_abstract_skipped(self):
        data = {"data": [{"citingPaper": {"corpusId": "cid1", "title": "  ", "abstract": "  "}}]}
        assert _parse_citations_response(data) == []

    def test_multiple_citing_papers(self):
        data = _make_citations_data(
            ("cid1", "Paper A", "Abstract A.", 2022),
            ("cid2", "Paper B", "Abstract B.", 2023),
        )
        result = _parse_citations_response(data)
        assert len(result) == 2
        urls = {r["url"] for r in result}
        assert "https://www.semanticscholar.org/paper/cid1" in urls
        assert "https://www.semanticscholar.org/paper/cid2" in urls

    def test_numeric_corpus_id_in_url(self):
        data = {"data": [{"citingPaper": {"corpusId": 99999, "title": "A Paper", "abstract": "Abstract."}}]}
        result = _parse_citations_response(data)
        assert result[0]["url"] == "https://www.semanticscholar.org/paper/99999"

    @pytest.mark.parametrize(
        "corpus_id, expected_url",
        [
            ("simple", "https://www.semanticscholar.org/paper/simple"),
            ("203952528", "https://www.semanticscholar.org/paper/203952528"),
        ],
        ids=["string-id", "numeric-string-id"],
    )
    def test_url_construction(self, corpus_id, expected_url):
        data = _make_citations_data((corpus_id, "Title", "Abstract.", None))
        result = _parse_citations_response(data)
        assert result[0]["url"] == expected_url


# ---------------------------------------------------------------------------
# CitationGraphAgent._chunk
# ---------------------------------------------------------------------------


class TestCitationGraphAgentChunk:
    def test_chunk_passes_raw_through_unchanged(self):
        ag = _make_agent()
        sq = SubQuery(query="test")
        chunks = [
            RawChunk(
                url="https://www.semanticscholar.org/paper/x",
                text="Citing paper text.",
                retrieved_at="2024-01-01T00:00:00+00:00",
                source_type=CITATION,
                sub_query=sq.query,
            )
        ]
        assert ag._chunk(chunks) is chunks

    def test_chunk_empty_list(self):
        ag = _make_agent()
        assert ag._chunk([]) == []


# ---------------------------------------------------------------------------
# CitationGraphAgent._search
# ---------------------------------------------------------------------------


class TestCitationGraphAgentSearch:
    async def test_happy_path_returns_raw_chunks(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response(
            _make_citations_data(("cite1", "Citing Paper", "Builds on the seed work.", 2024))
        )
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="attention mechanisms"))

        assert len(result) == 1
        assert isinstance(result[0], RawChunk)
        assert result[0].url == "https://www.semanticscholar.org/paper/cite1"

    async def test_source_type_is_citation(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response(_make_citations_data(("c1", "Title", "Abstract.", 2024)))
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result[0].source_type == CITATION

    async def test_sub_query_field_set_on_chunks(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response(_make_citations_data(("c1", "Title", "Abstract.", 2024)))
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        query = "transformer attention mechanisms"
        result = await _make_agent()._search(SubQuery(query=query))
        assert result[0].sub_query == query

    async def test_phase1_no_results_returns_empty_without_phase2_call(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data(None))
        mock = _mock_client([phase1])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="very obscure query"))

        assert result == []
        assert mock.get.call_count == 1  # Phase 2 must NOT be called

    async def test_phase1_http_error_propagates(self, mocker):
        phase1 = _make_http_response({}, status_code=429)
        mock = _mock_client([phase1])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_phase2_http_error_propagates(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response({}, status_code=500)
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_phase2_no_citations_returns_empty(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response({"data": []})
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_api_key_sent_in_both_phases(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response(_make_citations_data(("c1", "T", "A.", 2024)))
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        await _make_agent()._search(SubQuery(query="test"))

        for call in mock.get.call_args_list:
            _, kwargs = call
            assert kwargs.get("headers", {}).get("x-api-key") == "s2-key"

    async def test_phase2_limit_matches_agent_config(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response({"data": []})
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        await _make_agent(max_results=7)._search(SubQuery(query="test"))

        _, kwargs = mock.get.call_args_list[1]
        assert kwargs.get("params", {}).get("limit") == 7

    async def test_phase1_query_param_matches_sub_query(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response({"data": []})
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        await _make_agent()._search(SubQuery(query="my specific research query"))

        _, kwargs = mock.get.call_args_list[0]
        assert kwargs.get("params", {}).get("query") == "my specific research query"

    async def test_phase2_url_contains_seed_corpus_id(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("the_seed_corpus_id"))
        phase2 = _make_http_response({"data": []})
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        await _make_agent()._search(SubQuery(query="test"))

        args, _ = mock.get.call_args_list[1]
        assert "CorpusId:the_seed_corpus_id" in args[0]

    async def test_multiple_citations_all_returned(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response(
            _make_citations_data(
                ("c1", "Paper One", "Abstract one.", 2022),
                ("c2", "Paper Two", "Abstract two.", 2023),
                ("c3", "Paper Three", "Abstract three.", 2024),
            )
        )
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert len(result) == 3

    async def test_retrieved_at_is_iso_string(self, mocker):
        phase1 = _make_http_response(_make_paper_search_data("seed_id"))
        phase2 = _make_http_response(_make_citations_data(("c1", "T", "A.", 2024)))
        mock = _mock_client([phase1, phase2])
        mocker.patch("mara.agents.citation_graph.agent.httpx.AsyncClient", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        # ISO 8601 format check: must contain 'T' separator and timezone info
        assert "T" in result[0].retrieved_at
        assert "+" in result[0].retrieved_at or "Z" in result[0].retrieved_at


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestCitationGraphRegistration:
    def test_citation_graph_is_registered(self):
        import mara.agents.registry as reg_mod

        assert "citation_graph" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["citation_graph"].cls is CitationGraphAgent

    def test_default_config_rate_limit_rps(self):
        import mara.agents.registry as reg_mod

        reg = reg_mod._REGISTRY["citation_graph"]
        assert reg.config.rate_limit_rps == 1.0

    def test_default_config_max_results(self):
        import mara.agents.registry as reg_mod

        reg = reg_mod._REGISTRY["citation_graph"]
        assert reg.config.max_results == 20
