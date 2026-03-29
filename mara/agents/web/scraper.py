"""Firecrawl scraping helpers for the web agent."""

from __future__ import annotations

import asyncio
import logging

from firecrawl import FirecrawlApp

_log = logging.getLogger(__name__)


def _extract_text(result: object) -> str | None:
    """Extract markdown or content text from a Firecrawl scrape result.

    Accepts both dict-style results and object-style ``ScrapeResponse``.
    Returns stripped text, or ``None`` when no usable text is found.
    """
    if result is None:
        return None

    # Prefer markdown; fall back to content.
    if isinstance(result, dict):
        markdown = result.get("markdown") or ""
        if markdown.strip():
            return markdown.strip()
        content = result.get("content") or ""
        if content.strip():
            return content.strip()
        return None

    # Object-style response (e.g. ScrapeResponse from newer SDK versions)
    markdown = getattr(result, "markdown", None) or ""
    if markdown.strip():
        return markdown.strip()
    content = getattr(result, "content", None) or ""
    if content.strip():
        return content.strip()
    return None


async def scrape_url(url: str, api_key: str, timeout: float) -> str | None:
    """Scrape *url* with Firecrawl and return the markdown text.

    Runs the synchronous Firecrawl SDK call in a thread pool via
    ``asyncio.to_thread`` so it doesn't block the event loop.

    Returns ``None`` when scraping fails or produces no text.
    """
    try:
        app = FirecrawlApp(api_key=api_key)
        result = await asyncio.to_thread(app.scrape, url, formats=["markdown"])
        return _extract_text(result)
    except Exception as exc:
        _log.debug("Firecrawl scrape failed for %s: %s", url, exc)
        return None
