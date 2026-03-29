"""Tests for mara.agents.pubmed.agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.pubmed.agent import (
    ABSTRACT_ONLY,
    PMC_XML,
    PubMedAgent,
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


def _make_agent(**cfg_overrides) -> PubMedAgent:
    return PubMedAgent(config=_config(**cfg_overrides))


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


def _xml_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = {}
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_esearch(pmids: list[str]) -> dict:
    return {"esearchresult": {"idlist": pmids}}


def _make_esummary(items: list[dict]) -> dict:
    result: dict = {"uids": [i["uid"] for i in items]}
    for item in items:
        result[item["uid"]] = item
    return {"result": result}


def _make_esummary_item(uid: str, title: str = "Title", pmc_id: str | None = None) -> dict:
    article_ids = []
    if pmc_id:
        article_ids.append({"idtype": "pmc", "value": pmc_id})
    return {"uid": uid, "title": title, "articleids": article_ids}


_PMC_XML = (
    "<article><body>"
    "<sec><title>Methods</title><p>We used X.</p></sec>"
    "</body></article>"
)

_ABSTRACT_XML = (
    "<PubmedArticle><Abstract>"
    "<AbstractText>This is the abstract text.</AbstractText>"
    "</Abstract></PubmedArticle>"
)


def _mock_client(responses: list[MagicMock]) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# PubMedAgent._chunk
# ---------------------------------------------------------------------------


class TestPubMedAgentChunk:
    def test_chunk_returns_raw_unchanged(self):
        agent = _make_agent()
        sq = SubQuery(query="test")
        chunks = [
            RawChunk(
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                text="Some text.",
                retrieved_at="2024-01-01T00:00:00+00:00",
                source_type=PMC_XML,
                sub_query=sq.query,
            )
        ]
        assert agent._chunk(chunks) is chunks


# ---------------------------------------------------------------------------
# PubMedAgent._search
# ---------------------------------------------------------------------------


class TestPubMedAgentSearch:
    async def test_happy_path_pmc_returns_raw_chunks(self, mocker):
        esearch = _json_response(_make_esearch(["12345"]))
        esummary = _json_response(
            {"result": {**_make_esummary([_make_esummary_item("12345", pmc_id="PMC123")])["result"]}}
        )
        efetch = _xml_response(_PMC_XML)
        client = _mock_client([esearch, esummary, efetch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) >= 1
        assert all(isinstance(c, RawChunk) for c in result)
        assert result[0].source_type == PMC_XML

    async def test_happy_path_abstract_fallback(self, mocker):
        esearch = _json_response(_make_esearch(["99999"]))
        esummary = _json_response(
            {"result": {**_make_esummary([_make_esummary_item("99999")])["result"]}}
        )
        efetch = _xml_response(_ABSTRACT_XML)
        client = _mock_client([esearch, esummary, efetch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))

        assert len(result) == 1
        assert result[0].source_type == ABSTRACT_ONLY
        assert result[0].text == "This is the abstract text."

    async def test_source_type_pmc_xml(self, mocker):
        esearch = _json_response(_make_esearch(["12345"]))
        esummary = _json_response(
            {"result": {**_make_esummary([_make_esummary_item("12345", pmc_id="PMC123")])["result"]}}
        )
        efetch = _xml_response(_PMC_XML)
        client = _mock_client([esearch, esummary, efetch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert all(c.source_type == PMC_XML for c in result)

    async def test_sub_query_field_set(self, mocker):
        esearch = _json_response(_make_esearch(["12345"]))
        esummary = _json_response(
            {"result": {**_make_esummary([_make_esummary_item("12345")])["result"]}}
        )
        efetch = _xml_response(_ABSTRACT_XML)
        client = _mock_client([esearch, esummary, efetch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        query = "machine learning"
        result = await _make_agent()._search(SubQuery(query=query))
        assert all(c.sub_query == query for c in result)

    async def test_url_contains_pmid(self, mocker):
        esearch = _json_response(_make_esearch(["42"]))
        esummary = _json_response(
            {"result": {**_make_esummary([_make_esummary_item("42")])["result"]}}
        )
        efetch = _xml_response(_ABSTRACT_XML)
        client = _mock_client([esearch, esummary, efetch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result[0].url == "https://pubmed.ncbi.nlm.nih.gov/42/"

    async def test_empty_idlist_returns_empty(self, mocker):
        esearch = _json_response(_make_esearch([]))
        client = _mock_client([esearch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_api_key_sent_in_esearch(self, mocker):
        esearch = _json_response(_make_esearch([]))
        client = _mock_client([esearch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        await _make_agent()._search(SubQuery(query="test"))

        _, kwargs = client.get.call_args_list[0]
        assert kwargs.get("params", {}).get("api_key") == "ncbi-key"

    async def test_api_key_omitted_when_empty(self, mocker):
        esearch = _json_response(_make_esearch([]))
        client = _mock_client([esearch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        await _make_agent(ncbi_api_key="")._search(SubQuery(query="test"))

        _, kwargs = client.get.call_args_list[0]
        assert "api_key" not in kwargs.get("params", {})

    async def test_retmax_matches_config(self, mocker):
        esearch = _json_response(_make_esearch([]))
        client = _mock_client([esearch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        await _make_agent(pubmed_max_results=5)._search(SubQuery(query="test"))

        _, kwargs = client.get.call_args_list[0]
        assert kwargs.get("params", {}).get("retmax") == 5

    async def test_sleep_called_with_correct_delay(self, mocker):
        esearch = _json_response(_make_esearch([]))
        client = _mock_client([esearch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mock_sleep = mocker.patch(
            "mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock
        )

        await _make_agent(pubmed_rate_limit_per_second=4.0)._search(SubQuery(query="test"))

        mock_sleep.assert_called_with(pytest.approx(0.25))

    async def test_http_error_on_esearch_propagates(self, mocker):
        import httpx

        esearch = _json_response({}, status_code=500)
        client = _mock_client([esearch])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        with pytest.raises(httpx.HTTPStatusError):
            await _make_agent()._search(SubQuery(query="test"))

    async def test_article_with_bad_metadata_skipped(self, mocker):
        """If esummary item has no title, parse_article_metadata returns None → skip."""
        esearch = _json_response(_make_esearch(["99"]))
        # Item with no title → will be skipped
        bad_item = {"uid": "99", "articleids": []}
        esummary = _json_response({"result": {"uids": ["99"], "99": bad_item}})
        client = _mock_client([esearch, esummary])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []

    async def test_multiple_papers_multiple_chunks(self, mocker):
        esearch = _json_response(_make_esearch(["1", "2"]))
        esummary = _json_response(
            {
                "result": {
                    **_make_esummary(
                        [
                            _make_esummary_item("1"),
                            _make_esummary_item("2"),
                        ]
                    )["result"]
                }
            }
        )
        efetch1 = _xml_response(_ABSTRACT_XML)
        efetch2 = _xml_response(_ABSTRACT_XML)
        client = _mock_client([esearch, esummary, efetch1, efetch2])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert len(result) == 2

    async def test_pmc_empty_sections_falls_through_to_abstract(self, mocker):
        """If PMC efetch returns XML with no parseable sections, skip that article."""
        esearch = _json_response(_make_esearch(["12345"]))
        esummary = _json_response(
            {"result": {**_make_esummary([_make_esummary_item("12345", pmc_id="PMC123")])["result"]}}
        )
        # PMC XML with no <sec> elements → parse_pmc_sections returns []
        empty_pmc_xml = _xml_response("<article><body></body></article>")
        client = _mock_client([esearch, esummary, empty_pmc_xml])
        mocker.patch("mara.agents.pubmed.agent.httpx.AsyncClient", return_value=client)
        mocker.patch("mara.agents.pubmed.agent.asyncio.sleep", new_callable=AsyncMock)

        result = await _make_agent()._search(SubQuery(query="test"))
        assert result == []


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPubMedRegistration:
    def test_pubmed_registered(self):
        import mara.agents.registry as reg_mod

        assert "pubmed" in reg_mod._REGISTRY
        assert reg_mod._REGISTRY["pubmed"] is PubMedAgent
