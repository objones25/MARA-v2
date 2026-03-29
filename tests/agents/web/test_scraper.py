"""Tests for mara.agents.web.scraper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agents.web.scraper import _extract_text, scrape_url


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_dict_with_markdown(self):
        result = {"markdown": "# Title\n\nBody text.", "metadata": {}}
        assert _extract_text(result) == "# Title\n\nBody text."

    def test_dict_with_content_fallback(self):
        result = {"content": "Plain content here."}
        assert _extract_text(result) == "Plain content here."

    def test_dict_markdown_takes_precedence_over_content(self):
        result = {"markdown": "markdown text", "content": "content text"}
        assert _extract_text(result) == "markdown text"

    def test_object_with_markdown_attribute(self):
        obj = MagicMock()
        obj.markdown = "Object markdown."
        obj.content = "Object content."
        assert _extract_text(obj) == "Object markdown."

    def test_object_with_content_attribute_only(self):
        obj = MagicMock(spec=["content"])
        obj.content = "Only content."
        assert _extract_text(obj) == "Only content."

    def test_none_returns_none(self):
        assert _extract_text(None) is None

    def test_whitespace_only_markdown_returns_none(self):
        result = {"markdown": "   \n\t  "}
        assert _extract_text(result) == None

    def test_empty_markdown_falls_to_content(self):
        result = {"markdown": "", "content": "Fallback content."}
        assert _extract_text(result) == "Fallback content."

    def test_both_empty_returns_none(self):
        result = {"markdown": "", "content": ""}
        assert _extract_text(result) is None

    def test_strips_leading_trailing_whitespace(self):
        result = {"markdown": "  text  "}
        assert _extract_text(result) == "text"

    def test_dict_missing_both_keys_returns_none(self):
        assert _extract_text({"html": "<p>hello</p>"}) is None

    def test_object_with_empty_markdown_and_content_returns_none(self):
        obj = MagicMock()
        obj.markdown = ""
        obj.content = ""
        assert _extract_text(obj) is None


# ---------------------------------------------------------------------------
# scrape_url
# ---------------------------------------------------------------------------


class TestScrapeUrl:
    async def test_returns_markdown_text_on_success(self):
        mock_app = MagicMock()
        mock_app.scrape.return_value = {"markdown": "# Article\n\nContent here."}

        with patch("mara.agents.web.scraper.FirecrawlApp", return_value=mock_app):
            result = await scrape_url("https://example.com", "fc-key", 30.0)

        assert result == "# Article\n\nContent here."
        mock_app.scrape.assert_called_once_with(
            "https://example.com", formats=["markdown"]
        )

    async def test_returns_none_on_firecrawl_exception(self):
        mock_app = MagicMock()
        mock_app.scrape.side_effect = RuntimeError("scrape failed")

        with patch("mara.agents.web.scraper.FirecrawlApp", return_value=mock_app):
            result = await scrape_url("https://example.com", "fc-key", 30.0)

        assert result is None

    async def test_returns_none_when_no_text_extracted(self):
        mock_app = MagicMock()
        mock_app.scrape.return_value = {"markdown": "", "content": ""}

        with patch("mara.agents.web.scraper.FirecrawlApp", return_value=mock_app):
            result = await scrape_url("https://example.com", "fc-key", 30.0)

        assert result is None

    async def test_uses_api_key_for_firecrawl(self):
        mock_app = MagicMock()
        mock_app.scrape.return_value = {"markdown": "text"}

        with patch("mara.agents.web.scraper.FirecrawlApp", return_value=mock_app) as mock_cls:
            await scrape_url("https://example.com", "my-api-key", 30.0)

        mock_cls.assert_called_once_with(api_key="my-api-key")

    async def test_runs_in_thread(self):
        """Verify scrape_url uses asyncio.to_thread (sync SDK in async context)."""
        mock_app = MagicMock()
        mock_app.scrape.return_value = {"markdown": "content"}

        with patch("mara.agents.web.scraper.FirecrawlApp", return_value=mock_app):
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = {"markdown": "content"}
                result = await scrape_url("https://example.com", "key", 30.0)

        assert mock_thread.called

    async def test_firecrawl_constructor_raises(self):
        with patch(
            "mara.agents.web.scraper.FirecrawlApp",
            side_effect=ValueError("bad key"),
        ):
            result = await scrape_url("https://example.com", "bad", 30.0)

        assert result is None
