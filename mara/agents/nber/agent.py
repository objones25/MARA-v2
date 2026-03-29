"""NBER Working Papers specialist agent."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from curl_cffi.requests import AsyncSession

from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

WORKING_PAPER = "working_paper"

_NBER_BASE = "https://www.nber.org"
_NBER_API = f"{_NBER_BASE}/api/v1/working_page_listing/contentType/working_paper/_/_/search"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove all HTML tags from *text*."""
    return _TAG_RE.sub("", text)


def _parse_paper_results(data: list) -> list[dict]:
    """Parse a list of NBER paper dicts into url+text dicts.

    Each dict must have a ``url`` (relative NBER path) and at least one of
    ``title`` or ``abstract``.  Papers failing either requirement are skipped.
    """
    out = []
    for item in data:
        url_path = (item.get("url") or "").strip()
        if not url_path:
            continue

        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        if not title and not abstract:
            continue

        displaydate = (item.get("displaydate") or "").strip()
        authors_raw: list = item.get("authors") or []
        author_names = [_strip_html(a).strip() for a in authors_raw if a]
        author_names = [n for n in author_names if n]

        parts: list[str] = []
        if title:
            parts.append(f"{displaydate}: {title}" if displaydate else title)
        if author_names:
            parts.append(f"Authors: {', '.join(author_names)}")
        if abstract:
            parts.append(abstract)

        out.append({
            "url": f"{_NBER_BASE}{url_path}",
            "text": "\n\n".join(parts),
        })
    return out


@agent(
    "nber",
    description="NBER — retrieves National Bureau of Economic Research working papers on economics.",
    capabilities=[
        "Search across thousands of NBER working papers on economics and finance",
        "Returns title, abstract, authors, and publication date for each paper",
        "Covers macroeconomics, labor economics, public finance, health economics, and more",
        "Provides direct NBER URLs for each paper",
    ],
    limitations=[
        "Working papers only — not peer-reviewed journal articles",
        "Abstracts may be truncated in API responses; full text requires separate access",
        "Coverage is limited to papers indexed by NBER (economics focus)",
        "Poor fit for non-economics topics (biology, computer science, medicine, etc.)",
    ],
    example_queries=[
        "effects of monetary policy on inflation and output",
        "minimum wage employment effects labor market",
        "carbon tax climate policy macroeconomic impact",
        "income inequality wealth distribution United States",
    ],
    config=AgentConfig(max_results=20, rate_limit_rps=1.0),
)
class NBERAgent(SpecialistAgent):
    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        query = sub_query.query
        retrieved_at = datetime.now(timezone.utc).isoformat()

        async with AsyncSession(impersonate="chrome120") as session:
            resp = await session.get(
                _NBER_API,
                params={"q": query, "page": 1, "perPage": self.agent_config.max_results},
            )
            resp.raise_for_status()

        papers = _parse_paper_results(resp.json().get("results", []))
        return [
            RawChunk(
                url=p["url"],
                text=p["text"],
                retrieved_at=retrieved_at,
                source_type=WORKING_PAPER,
                sub_query=query,
            )
            for p in papers
        ]
