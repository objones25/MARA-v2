"""ArXiv specialist agent.

Discovers papers via the ArXiv Atom API and retrieves content via the
four-level fallback chain (LaTeX → PDF in tarball → rendered PDF → abstract).
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from mara.agents.arxiv.fetcher import (
    ABSTRACT_ONLY,  # noqa: F401 – re-exported for tests / type-checking
    LATEX,
    PDF_FROM_TARBALL,  # noqa: F401
    PDF_RENDERED,  # noqa: F401
    _now_iso,
    chunks_from_abstract,
    chunks_from_latex,
    chunks_from_pdf,
    extract_from_tarball,
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
        return url[idx + len(marker) :]
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
        "Full LaTeX source for most CS/physics/math papers",
        "PDF fallback when LaTeX is unavailable",
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
    config=AgentConfig(max_concurrent=1, retry_backoff_base=3.0),
)
class ArxivAgent(SpecialistAgent):
    """Retrieves research papers from ArXiv.

    Discovery uses the export.arxiv.org Atom API.  Content follows the
    four-level fallback chain defined in ``fetcher.py``.

    The ``_chunk()`` override passes LATEX-sourced chunks straight through
    (they are already section-sized semantic units) and applies the
    sliding-window splitter only to PDF and abstract chunks.
    """

    # Gate every _search() entry: 3 s between discovery calls across all
    # ArxivAgent instances so concurrent sub-query runs respect ArXiv's limits.
    _rate_limit_interval: float = _RATE_LIMIT_DELAY

    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        """Bypass sliding window for LaTeX sections; delegate the rest to super."""
        latex = [c for c in raw if c.source_type == LATEX]
        other = [c for c in raw if c.source_type != LATEX]
        return latex + (super()._chunk(other) if other else [])

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Discover and retrieve ArXiv papers for *sub_query*.

        Raises:
            httpx.HTTPError: if the ArXiv API discovery call fails.
        """
        headers = {"User-Agent": "MARA-research-agent/0.1 (https://github.com/mara; mailto:contact@mara.local)"}
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            # Build the query string manually so the colon in "all:<term>"
            # is NOT percent-encoded (ArXiv rejects %3A).
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
            for i, entry in enumerate(entries):
                if i > 0:
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
        """Retrieve content for one paper using the fallback chain.

        Per-paper failures are logged at DEBUG and fall through to the next
        level rather than raising.
        """
        versioned_id: str = entry["versioned_id"]
        canonical_url: str = entry["canonical_url"]
        abstract: str = entry["abstract"]
        pdf_url: str = entry["pdf_url"]
        retrieved_at = _now_iso()

        # Step 1 — source tarball
        tarball = await fetch_source_tarball(client, versioned_id)
        if tarball is not None:
            tex_contents, tarball_pdf = extract_from_tarball(tarball)

            # Step 1a — LaTeX source files present
            if tex_contents:
                latex_chunks = chunks_from_latex(
                    tex_contents, canonical_url, query, retrieved_at
                )
                if latex_chunks:
                    return latex_chunks
                # LaTeX files were present but produced no usable text;
                # fall through to the PDF-in-tarball path.
                _log.debug("latex produced no chunks for %s; trying tarball PDF", versioned_id)

            # Step 2 — PDF embedded in the tarball
            if tarball_pdf is not None:
                pdf_chunks = chunks_from_pdf(
                    tarball_pdf, canonical_url, query, retrieved_at, PDF_FROM_TARBALL
                )
                if pdf_chunks:
                    return pdf_chunks
                _log.debug("tarball PDF produced no text for %s", versioned_id)

        # Step 3 — rendered (versioned) PDF URL
        try:
            pdf_resp = await client.get(pdf_url, follow_redirects=True)
            if pdf_resp.status_code == 200:
                pdf_chunks = chunks_from_pdf(
                    bytes(pdf_resp.content),
                    canonical_url,
                    query,
                    retrieved_at,
                    PDF_RENDERED,
                )
                if pdf_chunks:
                    return pdf_chunks
                _log.debug("rendered PDF produced no text for %s", versioned_id)
        except Exception as exc:
            _log.debug("rendered PDF fetch failed for %s: %s", versioned_id, exc)

        # Step 4 — abstract only
        if abstract:
            _log.debug("falling back to abstract for %s", versioned_id)
            return chunks_from_abstract(abstract, canonical_url, query, retrieved_at)

        _log.debug("no content retrieved for %s", versioned_id)
        return []
