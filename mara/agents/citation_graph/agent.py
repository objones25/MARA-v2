"""Semantic Scholar Citation Graph specialist agent."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

CITATION = "citation"

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_PAPER_URL = "https://www.semanticscholar.org/paper/{corpus_id}"


def _parse_paper_search_response(data: dict) -> str | None:
    items = data.get("data", [])
    if not items:
        return None
    corpus_id = items[0].get("corpusId")
    if corpus_id is None:
        return None
    return str(corpus_id)


def _parse_citations_response(data: dict) -> list[dict]:
    results = []
    for entry in data.get("data", []):
        paper = entry.get("citingPaper", {})
        corpus_id = paper.get("corpusId")
        if corpus_id is None:
            continue

        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        year = paper.get("year")

        if not title and not abstract:
            continue

        parts = []
        if title:
            parts.append(f"{year}: {title}" if year else title)
        if abstract:
            parts.append(abstract)

        text = "\n\n".join(parts)
        url = _S2_PAPER_URL.format(corpus_id=corpus_id)
        results.append({"url": url, "text": text})
    return results


@agent(
    "citation_graph",
    description="Walks the Semantic Scholar citation graph forward from a seed paper to find works that built upon it.",
    capabilities=[
        "Forward citation traversal (single hop) from a seed paper",
        "Surfaces papers that cite the most relevant work for a query",
        "Provides structured metadata: title, abstract, year, and S2 URL",
    ],
    limitations=[
        "Single-hop only — does not recurse into citations of citations",
        "Quality depends on Semantic Scholar coverage of the seed paper",
        "Requires a query specific enough to surface a useful seed paper",
    ],
    example_queries=[
        "What papers cite the original Transformer attention is all you need paper?",
        "Papers that built on AlphaFold's protein structure prediction approach",
        "Follow-up work to BERT language model pre-training",
    ],
    config=AgentConfig(rate_limit_rps=1.0, max_results=20),
)
class CitationGraphAgent(SpecialistAgent):
    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        headers = {"x-api-key": self.agent_config.api_key}
        retrieved_at = datetime.now(timezone.utc).isoformat()

        async with httpx.AsyncClient() as client:
            # Phase 1: find seed paper corpus ID
            phase1 = await client.get(
                f"{_S2_BASE}/paper/search",
                params={"query": sub_query.query, "limit": 1, "fields": "corpusId,title"},
                headers=headers,
            )
            phase1.raise_for_status()
            corpus_id = _parse_paper_search_response(phase1.json())
            if corpus_id is None:
                return []

            # Phase 2: fetch papers citing the seed
            # S2 requires "CorpusId:{id}" format in the path for corpus ID lookups
            phase2 = await client.get(
                f"{_S2_BASE}/paper/CorpusId:{corpus_id}/citations",
                params={
                    "limit": self.agent_config.max_results,
                    "fields": "corpusId,title,abstract,year",
                },
                headers=headers,
            )
            phase2.raise_for_status()
            citation_dicts = _parse_citations_response(phase2.json())

        return [
            RawChunk(
                url=c["url"],
                text=c["text"],
                retrieved_at=retrieved_at,
                source_type=CITATION,
                sub_query=sub_query.query,
            )
            for c in citation_dicts
        ]
