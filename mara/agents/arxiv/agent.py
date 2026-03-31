"""ArXiv specialist agent.

Discovers papers via the ArXiv Atom API, fetches the source tarball,
compiles main.tex with pdflatex, and extracts text from the resulting PDF.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from mara.agents.arxiv.fetcher import (
    LATEX,  # noqa: F401 – re-exported for tests / type-checking
    _now_iso,
    chunks_from_pdf,
    compile_latex_to_pdf,
    fetch_source_tarball,
)
from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

_log = logging.getLogger(__name__)

_ATOM_NS = "http://www.w3.org/2005/Atom"
_API_URL = "https://export.arxiv.org/api/query"
_RATE_LIMIT_DELAY = 3.0  # seconds between per-paper content requests


def _versioned_id_from_url(id_url: str) -> str:
    """Extract the versioned ArXiv ID from an entry ``<id>`` URL.

    Handles both new-style IDs (``2301.07041v2``) and old-style category
    prefixed IDs (``hep-ph/0201093v2``).

    Example::

        >>> _versioned_id_from_url("http://arxiv.org/abs/2301.07041v2")
        '2301.07041v2'
        >>> _versioned_id_from_url("http://arxiv.org/abs/hep-ph/0201093v2")
        'hep-ph/0201093v2'
    """
    url = id_url.rstrip("/")
    marker = "/abs/"
    idx = url.find(marker)
    if idx != -1:
        return url[idx + len(marker):]
    # Fallback: last path segment
    return url.split("/")[-1]


def _parse_feed(xml_text: str) -> list[dict]:
    """Parse an ArXiv Atom feed and return a list of entry dicts.

    Each dict contains:

    - ``versioned_id``: immutable version-qualified ID (e.g. ``"2301.07041v2"``)
    - ``canonical_url``: ``https://arxiv.org/abs/{versioned_id}``
    - ``abstract``: plain-text summary from the feed
    - ``pdf_url``: ``https://arxiv.org/pdf/{versioned_id}``

    Entries missing an ``<id>`` element are silently skipped.
    """
    root = ET.fromstring(xml_text)
    entries: list[dict] = []

    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        id_el = entry.find(f"{{{_ATOM_NS}}}id")
        if id_el is None or not id_el.text:
            continue

        versioned_id = _versioned_id_from_url(id_el.text.strip())
        canonical_url = f"https://arxiv.org/abs/{versioned_id}"

        title_el = entry.find(f"{{{_ATOM_NS}}}title")
        summary_el = entry.find(f"{{{_ATOM_NS}}}summary")

        entries.append(
            {
                "versioned_id": versioned_id,
                "canonical_url": canonical_url,
                "abstract": (
                    summary_el.text.strip()
                    if summary_el is not None and summary_el.text
                    else ""
                ),
                "title": (
                    title_el.text.strip()
                    if title_el is not None and title_el.text
                    else ""
                ),
                "pdf_url": f"https://arxiv.org/pdf/{versioned_id}",
            }
        )

    return entries


@agent(
    "arxiv",
    description="Retrieves preprints and papers from ArXiv covering physics, math, CS, and quantitative biology.",
    capabilities=[
        "Full compiled PDF text for most CS/physics/math papers",
        "Broad coverage of recent preprints before peer review",
        "Strong for theoretical, mathematical, and computational topics",
    ],
    limitations=[
        "No clinical or biomedical content (use pubmed)",
        "Preprints may not be peer-reviewed",
        "Rate-limited — avoid routing too many sub-queries here simultaneously",
        "Weak on social sciences, humanities, and applied industry research",
    ],
    example_queries=[
        "quantum error correction with surface codes",
        "transformer attention mechanism efficiency improvements",
        "reinforcement learning from human feedback theory",
        "topological phases of matter",
    ],
    config=AgentConfig(max_concurrent=1, retry_backoff_base=5.0, max_sub_queries=1, max_retries=5),
)
class ArxivAgent(SpecialistAgent):
    """Retrieves research papers from ArXiv.

    Discovery uses the export.arxiv.org Atom API.  Content is obtained by
    downloading the source tarball, compiling main.tex with pdflatex, and
    extracting text from the resulting PDF.
    """

    # Gate every _search() entry: 3 s between discovery calls across all
    # ArxivAgent instances so concurrent sub-query runs respect ArXiv's limits.
    _rate_limit_interval: float = _RATE_LIMIT_DELAY

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Discover and retrieve ArXiv papers for *sub_query*.

        Raises:
            httpx.HTTPError: if the ArXiv API discovery call fails.
        """
        headers = {"User-Agent": "MARA-research-agent/0.1 (https://github.com/mara; mailto:contact@mara.local)"}
        _log.debug("arxiv _search start t=%.3f", time.monotonic(), extra={"agent": "arxiv"})
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            qs = (
                f"search_query=all:{urllib.parse.quote(sub_query.query)}"
                f"&max_results={self.agent_config.max_results}"
                f"&sortBy=relevance"
            )
            resp = await client.get(f"{_API_URL}?{qs}")
            resp.raise_for_status()
            entries = _parse_feed(resp.text)
            _log.debug(
                "arxiv discovery: %d entry(ies) for %r",
                len(entries),
                sub_query.query[:60],
            )

            chunks: list[RawChunk] = []
            for entry in entries:
                await asyncio.sleep(_RATE_LIMIT_DELAY)
                paper_chunks = await self._fetch_paper(client, entry, sub_query.query)
                chunks.extend(paper_chunks)

        return chunks

    async def _fetch_paper(
        self,
        client: httpx.AsyncClient,
        entry: dict,
        query: str,
    ) -> list[RawChunk]:
        """Compile the LaTeX source for one paper and return text chunks."""
        versioned_id: str = entry["versioned_id"]
        canonical_url: str = entry["canonical_url"]
        retrieved_at = _now_iso()

        try:
            tarball = await fetch_source_tarball(client, versioned_id)
            if tarball is None:
                return []
            pdf_bytes = compile_latex_to_pdf(tarball)
            return chunks_from_pdf(pdf_bytes, canonical_url, query, retrieved_at)
        except Exception as exc:
            _log.debug("failed to retrieve %s: %s", versioned_id, exc, extra={"agent": "arxiv"})
            return []
