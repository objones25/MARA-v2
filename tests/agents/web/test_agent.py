"""Tests for mara.agents.web.agent — WebAgent and _extract_urls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.types import SubQuery
from mara.agents.web.agent import WebAgent, _extract_urls


# ---------------------------------------------------------------------------
# _extract_urls
# ---------------------------------------------------------------------------


class TestExtractUrls:
    def test_web_results(self):
        data = {"web": {"results": [{"url": "https://example.com"}, {"url": "https://other.com"}]}}
        assert _extract_urls(data) == ["https://example.com", "https://other.com"]

    def test_news_results(self):
        data = {"news": {"results": [{"url": "https://news.com/article"}]}}
        assert _extract_urls(data) == ["https://news.com/article"]

    def test_discussions_results_url_field(self):
        data = {"discussions": {"results": [{"url": "https://discuss.com/thread"}]}}
        assert "https://discuss.com/thread" in _extract_urls(data)

    def test_discussions_results_nested_data(self):
        data = {
            "discussions": {
                "results": [{"data": {"url": "https://discuss.com/nested"}}]
            }
        }
        assert "https://discuss.com/nested" in _extract_urls(data)

    def test_faq_results(self):
        data = {"faq": {"results": [{"url": "https://faq.example.com/q1"}]}}
        assert _extract_urls(data) == ["https://faq.example.com/q1"]

    def test_deduplicates_across_sections(self):
        data = {
            "web": {"results": [{"url": "https://example.com"}]},
            "news": {"results": [{"url": "https://example.com"}]},
        }
        result = _extract_urls(data)
        assert result.count("https://example.com") == 1

    def test_empty_response(self):
        assert _extract_urls({}) == []

    def test_missing_url_field_skipped(self):
        data = {"web": {"results": [{"title": "No URL here"}]}}
        assert _extract_urls(data) == []

    def test_empty_url_string_skipped(self):
        data = {"web": {"results": [{"url": ""}]}}
        assert _extract_urls(data) == []

    def test_all_sections_combined(self):
        data = {
            "web": {"results": [{"url": "https://web.com"}]},
            "news": {"results": [{"url": "https://news.com"}]},
            "discussions": {"results": [{"url": "https://discuss.com"}]},
            "faq": {"results": [{"url": "https://faq.com"}]},
        }
        result = _extract_urls(data)
        assert set(result) == {
            "https://web.com",
            "https://news.com",
            "https://discuss.com",
            "https://faq.com",
        }


# ---------------------------------------------------------------------------
# WebAgent._search
# ---------------------------------------------------------------------------


def _brave_response(urls: list[str]) -> dict:
    """Build a minimal Brave API JSON response with *urls* in web section."""
    return {"web": {"results": [{"url": u} for u in urls]}}


def _mock_http_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


class TestWebAgentSearch:
    async def test_basic_pipeline_returns_chunks(self, config, mocker):
        """Full pipeline: Brave → filter → scrape → RawChunks."""
        brave_data = _brave_response(["https://arxiv.org/abs/1", "https://example.com/x"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mocker.patch(
            "mara.agents.web.agent.scrape_url",
            new=AsyncMock(return_value="Scraped content."),
        )
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        chunks = await agent._search(SubQuery(query="quantum computing"))

        assert len(chunks) > 0
        assert all(c.source_type == "web" for c in chunks)
        assert all(c.sub_query == "quantum computing" for c in chunks)

    async def test_freshness_param_sent_when_configured(self, config, mocker):
        config = config.model_copy(update={"brave_freshness": "pw"})
        brave_data = _brave_response(["https://arxiv.org/abs/1"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mocker.patch("mara.agents.web.agent.scrape_url", new=AsyncMock(return_value=None))
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        await agent._search(SubQuery(query="news"))

        _, kwargs = mock_client.get.call_args
        assert kwargs.get("params", {}).get("freshness") == "pw"

    async def test_freshness_param_omitted_when_empty(self, config, mocker):
        brave_data = _brave_response(["https://arxiv.org/abs/1"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mocker.patch("mara.agents.web.agent.scrape_url", new=AsyncMock(return_value=None))
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        await agent._search(SubQuery(query="news"))

        _, kwargs = mock_client.get.call_args
        assert "freshness" not in kwargs.get("params", {})

    async def test_scrape_failure_produces_no_chunk(self, config, mocker):
        """A URL that Firecrawl cannot scrape is silently skipped."""
        brave_data = _brave_response(["https://example.com/x"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mocker.patch("mara.agents.web.agent.scrape_url", new=AsyncMock(return_value=None))
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []

    async def test_llm_ranking_skipped_when_disabled(self, config, mocker):
        config = config.model_copy(update={"web_llm_url_ranking": False})
        brave_data = _brave_response(["https://arxiv.org/abs/1"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mock_scrape = mocker.patch(
            "mara.agents.web.agent.scrape_url",
            new=AsyncMock(return_value="content"),
        )
        mock_rank = mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(),
        )

        agent = WebAgent(config)
        await agent._search(SubQuery(query="test"))

        mock_rank.assert_not_called()

    async def test_caps_scrape_urls_at_max(self, config, mocker):
        """Only web_max_scrape_urls URLs are scraped even if more pass the filter."""
        config = config.model_copy(update={"web_max_scrape_urls": 2})
        urls = [f"https://arxiv.org/abs/{i}" for i in range(5)]
        brave_data = _brave_response(urls)
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mock_scrape = mocker.patch(
            "mara.agents.web.agent.scrape_url",
            new=AsyncMock(return_value="content"),
        )
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        assert mock_scrape.call_count == 2
        assert len(chunks) == 2

    async def test_brave_http_error_propagates(self, config, mocker):
        """HTTPStatusError from Brave API is not swallowed."""
        import httpx

        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=error_resp)
        )
        error_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=error_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)

        agent = WebAgent(config)
        with pytest.raises(httpx.HTTPStatusError):
            await agent._search(SubQuery(query="test"))

    async def test_extra_snippets_param_sent(self, config, mocker):
        brave_data = _brave_response(["https://arxiv.org/abs/1"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mocker.patch("mara.agents.web.agent.scrape_url", new=AsyncMock(return_value=None))
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        await agent._search(SubQuery(query="test"))

        _, kwargs = mock_client.get.call_args
        assert kwargs.get("params", {}).get("extra_snippets") == "true"

    async def test_empty_brave_results_returns_no_chunks(self, config, mocker):
        brave_data = {}
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(side_effect=lambda urls, *a, **kw: urls),
        )

        agent = WebAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        assert chunks == []

    async def test_llm_ranking_skipped_when_no_urls_after_filter(self, config, mocker):
        """When all URLs are blocked, LLM ranking is not called."""
        brave_data = _brave_response(["https://www.facebook.com/page"])
        http_resp = _mock_http_response(brave_data)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=http_resp)

        mocker.patch("mara.agents.web.agent.httpx.AsyncClient", return_value=mock_client)
        mock_rank = mocker.patch(
            "mara.agents.web.agent.rank_urls_with_llm",
            new=AsyncMock(),
        )

        agent = WebAgent(config)
        chunks = await agent._search(SubQuery(query="test"))

        mock_rank.assert_not_called()
        assert chunks == []

    async def test_web_agent_registered(self):
        from mara.agents.registry import _REGISTRY

        assert "web" in _REGISTRY
        assert _REGISTRY["web"] is WebAgent
