"""Tests for mara.agents.web.url_filter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.web.url_filter import (
    DomainTier,
    classify_url,
    filter_urls_by_tier,
    rank_urls_with_llm,
)


# ---------------------------------------------------------------------------
# classify_url
# ---------------------------------------------------------------------------


class TestClassifyUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://www.facebook.com/page", DomainTier.BLOCK),
            ("https://twitter.com/user", DomainTier.BLOCK),
            ("https://x.com/user", DomainTier.BLOCK),
            ("https://www.instagram.com/post", DomainTier.BLOCK),
            ("https://www.reddit.com/r/science", DomainTier.BLOCK),
            ("https://www.youtube.com/watch", DomainTier.BLOCK),
            ("https://www.linkedin.com/in/person", DomainTier.BLOCK),
            ("https://www.tiktok.com/@user", DomainTier.BLOCK),
        ],
        ids=["facebook", "twitter", "x", "instagram", "reddit", "youtube", "linkedin", "tiktok"],
    )
    def test_block_tier(self, url, expected):
        assert classify_url(url) == expected

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://www.mit.edu/research/paper", DomainTier.PRIORITY),
            ("https://cdc.gov/data", DomainTier.PRIORITY),
            ("https://arxiv.org/abs/2301.07041", DomainTier.PRIORITY),
            ("https://pubmed.ncbi.nlm.nih.gov/12345", DomainTier.PRIORITY),
            ("https://www.nature.com/articles/123", DomainTier.PRIORITY),
            ("https://www.semanticscholar.org/paper/x", DomainTier.PRIORITY),
            ("https://ieee.org/paper", DomainTier.PRIORITY),
            ("https://www.ox.ac.uk/research", DomainTier.PRIORITY),
        ],
        ids=["edu", "gov", "arxiv", "pubmed", "nature", "semanticscholar", "ieee", "ac-uk"],
    )
    def test_priority_tier(self, url, expected):
        assert classify_url(url) == expected

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://en.wikipedia.org/wiki/Topic", DomainTier.GOOD),
            ("https://www.bbc.com/news/article", DomainTier.GOOD),
            ("https://www.reuters.com/story", DomainTier.GOOD),
            ("https://www.nytimes.com/article", DomainTier.GOOD),
            ("https://www.theguardian.com/world", DomainTier.GOOD),
            ("https://www.wired.com/story", DomainTier.GOOD),
        ],
        ids=["wikipedia", "bbc", "reuters", "nytimes", "guardian", "wired"],
    )
    def test_good_tier(self, url, expected):
        assert classify_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/article",
            "https://some-blog.io/post",
            "https://random-site.net/page",
        ],
        ids=["example", "blog-io", "random-net"],
    )
    def test_default_tier(self, url):
        assert classify_url(url) == DomainTier.DEFAULT

    def test_malformed_url_returns_default(self):
        assert classify_url("not-a-url") == DomainTier.DEFAULT

    def test_empty_string_returns_default(self):
        assert classify_url("") == DomainTier.DEFAULT


# ---------------------------------------------------------------------------
# filter_urls_by_tier
# ---------------------------------------------------------------------------


class TestFilterUrlsByTier:
    def test_removes_block_tier_urls(self):
        urls = [
            "https://www.reddit.com/r/science",
            "https://arxiv.org/abs/123",
        ]
        result = filter_urls_by_tier(urls)
        assert "https://www.reddit.com/r/science" not in result
        assert "https://arxiv.org/abs/123" in result

    def test_sorts_priority_before_good_before_default(self):
        urls = [
            "https://example.com/article",
            "https://en.wikipedia.org/wiki/Topic",
            "https://arxiv.org/abs/2301.07041",
        ]
        result = filter_urls_by_tier(urls)
        assert result[0] == "https://arxiv.org/abs/2301.07041"
        assert result[1] == "https://en.wikipedia.org/wiki/Topic"
        assert result[2] == "https://example.com/article"

    def test_all_blocked_returns_empty(self):
        urls = ["https://www.facebook.com/", "https://www.reddit.com/r/x"]
        assert filter_urls_by_tier(urls) == []

    def test_empty_input_returns_empty(self):
        assert filter_urls_by_tier([]) == []

    def test_preserves_relative_order_within_tier(self):
        urls = [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        ]
        result = filter_urls_by_tier(urls)
        assert result == urls  # all DEFAULT, order preserved

    def test_mixed_all_tiers(self):
        urls = [
            "https://www.tiktok.com/@user",
            "https://example.com/page",
            "https://www.bbc.com/news",
            "https://arxiv.org/abs/1",
        ]
        result = filter_urls_by_tier(urls)
        assert "https://www.tiktok.com/@user" not in result
        assert result[0] == "https://arxiv.org/abs/1"
        assert result[1] == "https://www.bbc.com/news"
        assert result[2] == "https://example.com/page"


# ---------------------------------------------------------------------------
# rank_urls_with_llm
# ---------------------------------------------------------------------------


class TestRankUrlsWithLlm:
    def _make_config(self):
        from mara.config import ResearchConfig
        return ResearchConfig(
            brave_api_key="brave-key",
            hf_token="hf-token",
            firecrawl_api_key="fc-key",
            core_api_key="core-key",
            s2_api_key="s2-key",
            ncbi_api_key="ncbi-key",
        )

    async def test_returns_valid_subset_from_llm(self):
        config = self._make_config()
        urls = [
            "https://arxiv.org/abs/1",
            "https://example.com/page",
            "https://www.bbc.com/news",
        ]
        llm_response = MagicMock()
        llm_response.content = "https://arxiv.org/abs/1\nhttps://www.bbc.com/news"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        with patch("mara.agents.web.url_filter.make_llm", return_value=mock_llm):
            result = await rank_urls_with_llm(urls, "quantum computing", config)

        assert result == ["https://arxiv.org/abs/1", "https://www.bbc.com/news"]

    async def test_discards_hallucinated_urls(self):
        config = self._make_config()
        urls = ["https://arxiv.org/abs/1", "https://example.com/page"]
        # LLM returns a URL not in the original set
        llm_response = MagicMock()
        llm_response.content = (
            "https://arxiv.org/abs/1\nhttps://hallucinated.com/fake"
        )

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        with patch("mara.agents.web.url_filter.make_llm", return_value=mock_llm):
            result = await rank_urls_with_llm(urls, "test query", config)

        assert "https://hallucinated.com/fake" not in result
        assert "https://arxiv.org/abs/1" in result

    async def test_falls_back_when_llm_returns_no_valid_urls(self):
        config = self._make_config()
        urls = ["https://arxiv.org/abs/1", "https://example.com/page"]
        llm_response = MagicMock()
        llm_response.content = "I cannot determine relevance."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        with patch("mara.agents.web.url_filter.make_llm", return_value=mock_llm):
            result = await rank_urls_with_llm(urls, "test query", config)

        assert result == urls

    async def test_falls_back_when_llm_raises(self):
        config = self._make_config()
        urls = ["https://arxiv.org/abs/1", "https://example.com/page"]

        with patch("mara.agents.web.url_filter.make_llm", side_effect=RuntimeError("API error")):
            result = await rank_urls_with_llm(urls, "test query", config)

        assert result == urls

    async def test_llm_invoke_raises_falls_back(self):
        config = self._make_config()
        urls = ["https://arxiv.org/abs/1"]

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch("mara.agents.web.url_filter.make_llm", return_value=mock_llm):
            result = await rank_urls_with_llm(urls, "query", config)

        assert result == urls

    async def test_ignores_blank_lines_in_response(self):
        config = self._make_config()
        urls = ["https://arxiv.org/abs/1", "https://example.com/page"]
        llm_response = MagicMock()
        llm_response.content = "\nhttps://arxiv.org/abs/1\n\n"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=llm_response)

        with patch("mara.agents.web.url_filter.make_llm", return_value=mock_llm):
            result = await rank_urls_with_llm(urls, "query", config)

        assert result == ["https://arxiv.org/abs/1"]
