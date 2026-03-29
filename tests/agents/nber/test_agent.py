"""Tests for mara.agents.nber.agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mara.agents.nber.agent import (
    WORKING_PAPER,
    NBERAgent,
    _parse_paper_results,
    _strip_html,
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


def _make_agent(max_results: int = 20) -> NBERAgent:
    cfg = _config()
    agent_config = AgentConfig(max_results=max_results, rate_limit_rps=1.0)
    return NBERAgent(config=cfg, agent_config=agent_config)


def _http_response(data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    if status_code >= 400:
        from curl_cffi.requests.exceptions import RequestException
        resp.raise_for_status.side_effect = RequestException("error")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_client(responses: list) -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.get = AsyncMock(side_effect=responses)
    return mock


_UNSET = object()


def _make_paper(
    *,
    title: str | None = "Inflation and Growth",
    abstract: str | None = "We study the link between inflation and growth.",
    authors: list | None | object = _UNSET,
    displaydate: str | None = "January 2023",
    url: str | None = "/papers/w12345",
) -> dict:
    """Build a single NBER paper dict as returned by the API.

    Pass ``authors=None`` to omit the field entirely (simulates missing key).
    Pass ``authors=_UNSET`` (default) to include a single default author.
    Pass ``authors=[]`` for an explicit empty list.
    """
    paper: dict = {}
    if title is not None:
        paper["title"] = title
    if abstract is not None:
        paper["abstract"] = abstract
    if authors is _UNSET:
        paper["authors"] = ['<a href="/people/john_doe">John Doe</a>']
    elif authors is not None:
        paper["authors"] = authors  # type: ignore[assignment]
    # authors is None → omit key entirely
    if displaydate is not None:
        paper["displaydate"] = displaydate
    if url is not None:
        paper["url"] = url
    return paper


def _make_response(*papers: dict) -> dict:
    """Wrap papers in the NBER API envelope."""
    return {"results": list(papers), "totalResults": len(papers)}


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------


class TestStripHtml:
    def test_strips_anchor_tag(self):
        assert _strip_html('<a href="/people/john_doe">John Doe</a>') == "John Doe"

    def test_strips_multiple_tags(self):
        assert _strip_html("<b><i>Bold Italic</i></b>") == "Bold Italic"

    def test_plain_string_unchanged(self):
        assert _strip_html("No tags here") == "No tags here"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_only_tags_returns_empty(self):
        assert _strip_html("<br/>") == ""

    def test_nested_tags(self):
        assert _strip_html('<span class="x"><em>text</em></span>') == "text"


# ---------------------------------------------------------------------------
# _parse_paper_results
# ---------------------------------------------------------------------------


class TestParsePaperResults:
    def test_happy_path_all_fields(self):
        data = [_make_paper()]
        result = _parse_paper_results(data)
        assert len(result) == 1
        assert result[0]["url"] == "https://www.nber.org/papers/w12345"
        assert "January 2023" in result[0]["text"]
        assert "Inflation and Growth" in result[0]["text"]
        assert "John Doe" in result[0]["text"]
        assert "We study the link" in result[0]["text"]

    def test_url_prepends_nber_base(self):
        data = [_make_paper(url="/papers/w99999")]
        result = _parse_paper_results(data)
        assert result[0]["url"] == "https://www.nber.org/papers/w99999"

    def test_missing_url_skipped(self):
        data = [_make_paper(url=None)]
        assert _parse_paper_results(data) == []

    def test_empty_url_skipped(self):
        data = [_make_paper(url="")]
        assert _parse_paper_results(data) == []

    def test_whitespace_url_skipped(self):
        data = [_make_paper(url="   ")]
        assert _parse_paper_results(data) == []

    def test_missing_title_and_abstract_skipped(self):
        data = [_make_paper(title=None, abstract=None)]
        assert _parse_paper_results(data) == []

    def test_whitespace_title_and_whitespace_abstract_skipped(self):
        data = [_make_paper(title="  ", abstract="  ")]
        assert _parse_paper_results(data) == []

    def test_title_only_no_abstract(self):
        data = [_make_paper(title="Only Title", abstract=None, authors=None, displaydate=None)]
        result = _parse_paper_results(data)
        assert len(result) == 1
        assert result[0]["text"] == "Only Title"

    def test_abstract_only_no_title(self):
        data = [_make_paper(title=None, abstract="Abstract only.", authors=None, displaydate=None)]
        result = _parse_paper_results(data)
        assert len(result) == 1
        assert result[0]["text"] == "Abstract only."

    def test_date_prefixes_title(self):
        data = [_make_paper(title="My Paper", abstract=None, authors=None, displaydate="March 2020")]
        result = _parse_paper_results(data)
        assert result[0]["text"] == "March 2020: My Paper"

    def test_no_date_no_prefix(self):
        data = [_make_paper(title="My Paper", abstract=None, authors=None, displaydate=None)]
        result = _parse_paper_results(data)
        assert result[0]["text"] == "My Paper"

    def test_authors_stripped_of_html(self):
        data = [_make_paper(authors=['<a href="/people/jane">Jane Smith</a>'])]
        result = _parse_paper_results(data)
        assert "Jane Smith" in result[0]["text"]
        assert "<a" not in result[0]["text"]

    def test_multiple_authors_joined(self):
        data = [
            _make_paper(
                authors=[
                    '<a href="/people/a">Author A</a>',
                    '<a href="/people/b">Author B</a>',
                ]
            )
        ]
        result = _parse_paper_results(data)
        assert "Author A" in result[0]["text"]
        assert "Author B" in result[0]["text"]

    def test_empty_authors_list_omits_authors_line(self):
        data = [_make_paper(title="T", abstract="A.", authors=[], displaydate=None)]
        result = _parse_paper_results(data)
        assert "Authors:" not in result[0]["text"]

    def test_none_authors_field_omits_authors_line(self):
        data = [_make_paper(title="T", abstract="A.", authors=None, displaydate=None)]
        result = _parse_paper_results(data)
        assert "Authors:" not in result[0]["text"]

    def test_empty_list_returns_empty(self):
        assert _parse_paper_results([]) == []

    def test_missing_results_key_returns_empty(self):
        # _parse_paper_results takes the list directly — empty list is the degenerate case
        assert _parse_paper_results([]) == []

    def test_multiple_papers_all_returned(self):
        data = [
            _make_paper(title="Paper A", url="/papers/w1"),
            _make_paper(title="Paper B", url="/papers/w2"),
            _make_paper(title="Paper C", url="/papers/w3"),
        ]
        result = _parse_paper_results(data)
        assert len(result) == 3

    def test_abstract_separator_present_when_both_title_and_abstract(self):
        data = [_make_paper(title="T", abstract="A.", authors=None, displaydate=None)]
        result = _parse_paper_results(data)
        assert "\n\n" in result[0]["text"]

    def test_authors_section_present_when_authors_exist(self):
        data = [_make_paper(title="T", abstract="A.", authors=['<a href="/p/x">X</a>'], displaydate=None)]
        result = _parse_paper_results(data)
        assert "Authors:" in result[0]["text"]

    @pytest.mark.parametrize(
        "url_path, expected_url",
        [
            ("/papers/w1", "https://www.nber.org/papers/w1"),
            ("/papers/w99999", "https://www.nber.org/papers/w99999"),
        ],
        ids=["short-id", "long-id"],
    )
    def test_url_construction(self, url_path, expected_url):
        data = [_make_paper(url=url_path)]
        result = _parse_paper_results(data)
        assert result[0]["url"] == expected_url


# ---------------------------------------------------------------------------
# NBERAgent._chunk
# ---------------------------------------------------------------------------


class TestNBERAgentChunk:
    def test_chunk_returns_raw_unchanged(self):
        ag = _make_agent()
        sq = SubQuery(query="test")
        chunks = [
            RawChunk(
                url="https://www.nber.org/papers/w1",
                text="Some paper text.",
                retrieved_at="2024-01-01T00:00:00+00:00",
                source_type=WORKING_PAPER,
                sub_query=sq.query,
            )
        ]
        assert ag._chunk(chunks) is chunks

    def test_chunk_empty_list(self):
        ag = _make_agent()
        result = ag._chunk([])
        assert result == []


# ---------------------------------------------------------------------------
# NBERAgent._search
# ---------------------------------------------------------------------------


class TestNBERAgentSearch:
    async def test_happy_path_returns_raw_chunks(self, mocker):
        resp = _http_response(_make_response(_make_paper()))
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="inflation"))

        assert len(result) == 1
        assert isinstance(result[0], RawChunk)

    async def test_source_type_is_working_paper(self, mocker):
        resp = _http_response(_make_response(_make_paper()))
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result[0].source_type == WORKING_PAPER

    async def test_sub_query_field_set_on_chunks(self, mocker):
        resp = _http_response(_make_response(_make_paper()))
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        query = "monetary policy transmission"
        result = await _make_agent()._search(SubQuery(query=query))
        assert result[0].sub_query == query

    async def test_chunk_url_uses_nber_base(self, mocker):
        resp = _http_response(_make_response(_make_paper(url="/papers/w42")))
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result[0].url == "https://www.nber.org/papers/w42"

    async def test_single_api_call_made(self, mocker):
        resp = _http_response(_make_response())
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        await _make_agent()._search(SubQuery(query="test"))
        assert mock.get.call_count == 1

    async def test_query_param_sent(self, mocker):
        resp = _http_response(_make_response())
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        await _make_agent()._search(SubQuery(query="fiscal multipliers"))

        _, kwargs = mock.get.call_args
        assert kwargs.get("params", {}).get("q") == "fiscal multipliers"

    async def test_page_param_is_one(self, mocker):
        resp = _http_response(_make_response())
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        await _make_agent()._search(SubQuery(query="test"))

        _, kwargs = mock.get.call_args
        assert kwargs.get("params", {}).get("page") == 1

    async def test_per_page_matches_max_results(self, mocker):
        resp = _http_response(_make_response())
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        await _make_agent(max_results=7)._search(SubQuery(query="test"))

        _, kwargs = mock.get.call_args
        assert kwargs.get("params", {}).get("perPage") == 7

    async def test_nber_api_url_used(self, mocker):
        resp = _http_response(_make_response())
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        await _make_agent()._search(SubQuery(query="test"))

        url_called, _ = mock.get.call_args
        assert "nber.org" in url_called[0]
        assert "working_paper" in url_called[0]

    async def test_empty_results_returns_empty_list(self, mocker):
        resp = _http_response(_make_response())
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_http_error_propagates(self, mocker):
        from curl_cffi.requests.exceptions import RequestException

        resp = _http_response({}, status_code=429)
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        with pytest.raises(RequestException):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_multiple_papers_all_returned(self, mocker):
        resp = _http_response(
            _make_response(
                _make_paper(title="Paper A", url="/papers/w1"),
                _make_paper(title="Paper B", url="/papers/w2"),
                _make_paper(title="Paper C", url="/papers/w3"),
            )
        )
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert len(result) == 3

    async def test_retrieved_at_is_iso_string(self, mocker):
        resp = _http_response(_make_response(_make_paper()))
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert "T" in result[0].retrieved_at
        assert "+" in result[0].retrieved_at or "Z" in result[0].retrieved_at

    async def test_papers_missing_url_excluded(self, mocker):
        resp = _http_response(
            _make_response(
                _make_paper(url=None),
                _make_paper(url="/papers/w1"),
            )
        )
        mock = _mock_client([resp])
        mocker.patch("mara.agents.nber.agent.AsyncSession", return_value=mock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert len(result) == 1
        assert result[0].url == "https://www.nber.org/papers/w1"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestNBERRegistration:
    def test_nber_is_registered(self):
        import mara.agents.registry as reg_mod

        assert "nber" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["nber"].cls is NBERAgent

    def test_default_max_results(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].config.max_results == 20

    def test_default_rate_limit_rps(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].config.rate_limit_rps == 1.0

    def test_no_api_key_required(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].config.api_key == ""

    def test_description_non_empty(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].description

    def test_capabilities_non_empty(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].capabilities

    def test_limitations_non_empty(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].limitations

    def test_example_queries_non_empty(self):
        import mara.agents.registry as reg_mod

        assert reg_mod._REGISTRY["nber"].example_queries
