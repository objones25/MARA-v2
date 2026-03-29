"""CORE specialist agent.

Discovers papers via the CORE API v3 and retrieves content following a
three-level fallback chain:

1. ``fullText`` field in the API response          (``source_type="fulltext"``)
2. PDF download via ``downloadUrl`` + pypdf         (``source_type="pdf_downloaded"``)
3. ``abstract`` field                               (``source_type="abstract_only"``)

Rate limiting uses a shared asyncio.Lock so concurrent pipeline runs respect
CORE's documented API rate limits.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.core.fetcher import extract_pdf_text
from mara.agents.registry import agent
from mara.agents.types import RawChunk, SubQuery

_log = logging.getLogger(__name__)

FULLTEXT = "fulltext"
PDF_DOWNLOADED = "pdf_downloaded"
ABSTRACT_ONLY = "abstract_only"

_CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"
_CORE_PAPER_URL = "https://core.ac.uk/display/{id}"

# Shared lock — all COREAgent instances share one lock so concurrent
# pipeline runs stay within CORE's API rate limits.
_CORE_LOCK: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _CORE_LOCK
    if _CORE_LOCK is None:
        _CORE_LOCK = asyncio.Lock()
    return _CORE_LOCK


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@agent("core")
class COREAgent(SpecialistAgent):
    """Retrieves research papers from the CORE open-access aggregator."""

    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        """CORE content is pre-chunked; pass through unchanged."""
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch papers from CORE for *sub_query*.

        Implements a three-level content fallback chain per paper:
        1. ``fullText`` field — used directly when non-empty.
        2. PDF download — fetched from ``downloadUrl``, text extracted via pypdf.
        3. ``abstract`` field — used when both fullText and PDF are unavailable.

        Papers with no content from any strategy are silently skipped.

        Raises:
            httpx.HTTPStatusError: if the CORE search API returns a non-2xx response.
        """
        lock = _get_lock()
        retrieved_at = _now_iso()
        headers = {"Authorization": f"Bearer {self.config.core_api_key}"}

        async with httpx.AsyncClient() as client:
            # Discover papers
            async with lock:
                search_resp = await client.get(
                    _CORE_SEARCH_URL,
                    params={"q": sub_query.query, "limit": self.config.core_max_results},
                    headers=headers,
                )
            search_resp.raise_for_status()

            results: list[dict] = search_resp.json().get("results", [])
            _log.debug(
                "core search: %d result(s) for %r", len(results), sub_query.query[:60]
            )

            chunks: list[RawChunk] = []
            for work in results:
                paper_id = work.get("id")
                if not paper_id:
                    continue
                url = _CORE_PAPER_URL.format(id=paper_id)

                # Strategy 1: fullText field
                full_text: str = work.get("fullText") or ""
                if full_text.strip():
                    chunks.append(
                        RawChunk(
                            url=url,
                            text=full_text.strip(),
                            retrieved_at=retrieved_at,
                            source_type=FULLTEXT,
                            sub_query=sub_query.query,
                        )
                    )
                    _log.debug("core: paper %s → fulltext (%d chars)", paper_id, len(full_text))
                    continue

                # Strategy 2: PDF download
                download_url: str = work.get("downloadUrl") or ""
                if download_url:
                    pdf_text = await _fetch_pdf_text(client, download_url)
                    if pdf_text:
                        chunks.append(
                            RawChunk(
                                url=url,
                                text=pdf_text,
                                retrieved_at=retrieved_at,
                                source_type=PDF_DOWNLOADED,
                                sub_query=sub_query.query,
                            )
                        )
                        _log.debug(
                            "core: paper %s → pdf_downloaded (%d chars)",
                            paper_id,
                            len(pdf_text),
                        )
                        continue

                # Strategy 3: abstract
                abstract: str = work.get("abstract") or ""
                if abstract.strip():
                    chunks.append(
                        RawChunk(
                            url=url,
                            text=abstract.strip(),
                            retrieved_at=retrieved_at,
                            source_type=ABSTRACT_ONLY,
                            sub_query=sub_query.query,
                        )
                    )
                    _log.debug(
                        "core: paper %s → abstract_only (%d chars)", paper_id, len(abstract)
                    )
                    continue

                _log.debug("core: skipping paper %s (no content)", paper_id)

        return chunks


async def _fetch_pdf_text(client: httpx.AsyncClient, download_url: str) -> str | None:
    """Download a PDF from *download_url* and extract its text.

    Returns extracted text, or ``None`` on any failure (non-200 status,
    extraction error, network exception).
    """
    try:
        resp = await client.get(download_url, follow_redirects=True)
        if resp.status_code != 200:
            _log.debug("core PDF fetch: HTTP %d for %s", resp.status_code, download_url)
            return None
        return extract_pdf_text(bytes(resp.content))
    except Exception as exc:
        _log.debug("core PDF fetch failed for %s: %s", download_url, exc)
        return None
