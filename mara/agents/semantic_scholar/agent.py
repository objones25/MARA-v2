"""Semantic Scholar specialist agent.

Discovers research snippets via the Semantic Scholar Snippet Search API.
Rate limiting (≤1 RPS by default, configurable via ``s2_max_rps``) is
handled by the ``SpecialistAgent`` base class.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

_log = logging.getLogger(__name__)

SNIPPET = "snippet"

_S2_BASE_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
_S2_PAPER_URL = "https://www.semanticscholar.org/paper/{paper_id}"


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


@agent(
    "s2",
    description="Retrieves citation-rich paper snippets from Semantic Scholar spanning all academic disciplines.",
    capabilities=[
        "Cross-disciplinary coverage including medicine, CS, social sciences, and engineering",
        "Pre-extracted relevant snippets — high signal-to-noise ratio",
        "Citation metadata useful for identifying influential papers",
        "Good for literature review and identifying key authors or institutions",
    ],
    limitations=[
        "Snippets only — no full text or LaTeX source",
        "Rate-limited to 1 RPS; avoid routing many sub-queries simultaneously",
        "Less coverage of very recent preprints compared to ArXiv",
        "Snippet context may be narrow if the paper is dense",
    ],
    example_queries=[
        "survey of large language model alignment techniques",
        "meta-analysis of CRISPR gene editing efficacy",
        "social network analysis of misinformation spread",
        "economic impact of automation on labor markets",
    ],
    config=AgentConfig(rate_limit_rps=1.0),
)
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
        retrieved_at = _now_iso()

        async with httpx.AsyncClient() as client:
            response = await client.get(
                _S2_BASE_URL,
                headers={"x-api-key": self.agent_config.api_key},
                params={
                    "query": sub_query.query,
                    "limit": self.agent_config.max_results,
                },
            )

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
