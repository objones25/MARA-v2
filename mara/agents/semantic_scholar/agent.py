"""Semantic Scholar specialist agent.

Discovers research snippets via the Semantic Scholar Snippet Search API and
rate-limits requests to ≤1 RPS (configurable) using a shared asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import agent
from mara.agents.types import RawChunk, SubQuery

_log = logging.getLogger(__name__)

SNIPPET = "snippet"

_S2_BASE_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
_S2_PAPER_URL = "https://www.semanticscholar.org/paper/{paper_id}"

# Shared lock — all SemanticScholarAgent instances share one lock so concurrent
# pipeline runs don't exceed the per-key rate limit.
_S2_LOCK: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _S2_LOCK
    if _S2_LOCK is None:
        _S2_LOCK = asyncio.Lock()
    return _S2_LOCK


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_snippet_response(data: dict) -> list[dict]:
    """Extract url/text pairs from a S2 snippet search response dict.

    Each item in ``data["data"]`` has a single ``snippet`` object (not a list)
    and identifies the paper by ``corpusId``.
    """
    results: list[dict] = []
    for item in data.get("data", []):
        paper = item.get("paper", {})
        corpus_id = paper.get("corpusId")
        if not corpus_id:
            continue
        url = _S2_PAPER_URL.format(paper_id=corpus_id)
        snippet = item.get("snippet", {})
        text = snippet.get("text", "").strip()
        if text:
            results.append({"url": url, "text": text})
    return results


@agent("s2")
class SemanticScholarAgent(SpecialistAgent):
    """Retrieves research snippets from the Semantic Scholar API."""

    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        """S2 snippets are pre-chunked; pass through unchanged."""
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch snippets from Semantic Scholar for *sub_query*.

        Raises:
            httpx.HTTPStatusError: if the API returns a non-2xx response.
        """
        lock = _get_lock()
        retrieved_at = _now_iso()

        async with httpx.AsyncClient() as client:
            async with lock:
                response = await client.get(
                    _S2_BASE_URL,
                    headers={"x-api-key": self.config.s2_api_key},
                    params={
                        "query": sub_query.query,
                        "limit": self.config.s2_max_results,
                    },
                )
                await asyncio.sleep(1.0 / self.config.s2_max_rps)

        response.raise_for_status()

        chunks: list[RawChunk] = []
        for item in _parse_snippet_response(response.json()):
            chunks.append(
                RawChunk(
                    url=item["url"],
                    text=item["text"],
                    retrieved_at=retrieved_at,
                    source_type=SNIPPET,
                    sub_query=sub_query.query,
                )
            )
        return chunks
