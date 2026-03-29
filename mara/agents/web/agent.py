"""Web specialist agent — Brave Search + URL filtering + Firecrawl scraping."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import agent
from mara.agents.types import RawChunk, SubQuery
from mara.agents.web.scraper import scrape_url
from mara.agents.web.url_filter import filter_urls_by_tier, rank_urls_with_llm

_log = logging.getLogger(__name__)

WEB = "web"

_BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_urls(data: dict) -> list[str]:
    """Extract unique URLs from a Brave Search API response.

    Collects from ``web``, ``news``, ``discussions``, and ``faq`` sections.
    Duplicate URLs (across sections) are dropped — first occurrence wins.
    """
    seen: set[str] = set()
    urls: list[str] = []

    def _add(url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    for r in data.get("web", {}).get("results", []):
        _add(r.get("url", ""))

    for r in data.get("news", {}).get("results", []):
        _add(r.get("url", ""))

    for r in data.get("discussions", {}).get("results", []):
        # Discussions may nest the URL inside a ``data`` sub-object.
        nested = r.get("data")
        if isinstance(nested, dict):
            _add(nested.get("url", ""))
        else:
            _add(r.get("url", ""))

    for r in data.get("faq", {}).get("results", []):
        _add(r.get("url", ""))

    return urls


@agent(
    "web",
    description="Retrieves live web content via Brave Search and Firecrawl scraping.",
    capabilities=[
        "Access to current events, news, and content not in academic databases",
        "Industry reports, blog posts, documentation, and technical tutorials",
        "Good for applied/commercial context around a research topic",
        "Discovers grey literature unavailable in academic repositories",
    ],
    limitations=[
        "No guarantee of peer-reviewed or scholarly content",
        "Content quality varies widely by domain",
        "Slower than academic APIs due to scraping overhead",
        "Poor fit for questions requiring primary research data or clinical evidence",
    ],
    example_queries=[
        "latest industry benchmarks for large language model inference speed",
        "current regulatory landscape for autonomous vehicles",
        "recent news on quantum computing commercialization",
        "open-source tools for federated learning deployment",
    ],
)
class WebAgent(SpecialistAgent):
    """Retrieves web content via Brave Search and Firecrawl scraping.

    Discovery: Brave Search API (web + news + discussions + faq sections).
    Filtering: two-pass — domain-tier regex then optional LLM ranking.
    Scraping: Firecrawl (sync SDK called via ``asyncio.to_thread``).
    """

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Discover, filter, and scrape URLs for *sub_query*.

        Raises:
            httpx.HTTPError: if the Brave API call fails.
        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.config.brave_api_key,
        }
        params: dict[str, str | int] = {
            "q": sub_query.query,
            "count": self.agent_config.max_results,
            "extra_snippets": "true",
        }
        if self.config.brave_freshness:
            params["freshness"] = self.config.brave_freshness

        async with httpx.AsyncClient(timeout=self.config.web_timeout_seconds) as client:
            resp = await client.get(_BRAVE_API_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        urls = _extract_urls(data)

        # Pass 1 — domain-tier regex filter
        urls = filter_urls_by_tier(urls)

        # Pass 2 — optional LLM relevance ranking (only when URLs remain)
        if self.config.web_llm_url_ranking and urls:
            urls = await rank_urls_with_llm(urls, sub_query.query, self.config)

        # Cap to configured maximum
        urls = urls[: self.config.web_max_scrape_urls]

        retrieved_at = _now_iso()
        chunks: list[RawChunk] = []

        for url in urls:
            text = await scrape_url(url, self.config.firecrawl_api_key, self.config.scrape_timeout_seconds)
            if text:
                chunks.append(
                    RawChunk(
                        url=url,
                        text=text,
                        retrieved_at=retrieved_at,
                        source_type=WEB,
                        sub_query=sub_query.query,
                    )
                )
            else:
                _log.debug("no text scraped for %s", url)

        return chunks
