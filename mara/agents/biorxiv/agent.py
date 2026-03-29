"""bioRxiv / medRxiv specialist agent."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

BIORXIV = "biorxiv"
MEDRXIV = "medrxiv"

_BIORXIV_BASE = "https://www.biorxiv.org"
_MEDRXIV_BASE = "https://www.medrxiv.org"
_API_BASE = "https://api.biorxiv.org"
_DATE_WINDOW_DAYS = 90

_SERVER_BASE = {
    BIORXIV: _BIORXIV_BASE,
    MEDRXIV: _MEDRXIV_BASE,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_biorxiv_response(data: dict) -> list[dict]:
    """Parse a bioRxiv/medRxiv API response into url+text+server dicts."""
    out = []
    for item in data.get("collection", []):
        doi = (item.get("doi") or "").strip()
        if not doi:
            continue

        abstract = (item.get("abstract") or "").strip()
        if not abstract:
            continue

        title = (item.get("title") or "").strip()
        text = f"{title}\n{abstract}" if title else abstract

        server = (item.get("server") or BIORXIV).lower()
        base = _SERVER_BASE.get(server, _BIORXIV_BASE)
        url = f"{base}/content/{doi}"

        out.append({"url": url, "text": text, "server": server})
    return out


def _keywords(query: str) -> list[str]:
    return [w.lower() for w in query.split() if w]


def _matches_query(item: dict, keywords: list[str]) -> bool:
    haystack = item["text"].lower()
    return any(kw in haystack for kw in keywords)


@agent(
    "biorxiv",
    description="bioRxiv / medRxiv — retrieves recent biology and medicine preprints.",
    capabilities=[
        "Latest preprints from bioRxiv (biology) and medRxiv (medicine/health sciences)",
        "Covers genomics, neuroscience, immunology, clinical trials, and epidemiology",
        "Useful for cutting-edge findings before formal peer-review publication",
        "Searches across titles and abstracts from both servers simultaneously",
    ],
    limitations=[
        "Preprints are not peer-reviewed — findings may change before final publication",
        "No keyword search API — uses date-window fetch with client-side filtering",
        "Limited to the most recent 90-day window; older papers not retrievable",
        "Poor fit for established results, engineering topics, or non-life-science research",
    ],
    example_queries=[
        "CRISPR gene editing off-target effects in human cells",
        "mRNA vaccine immune response mechanisms",
        "single-cell RNA sequencing of tumour microenvironment",
        "COVID-19 long-term neurological outcomes",
    ],
    config=AgentConfig(max_results=20),
)
class BioRxivAgent(SpecialistAgent):
    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        query = sub_query.query
        keywords = _keywords(query)
        retrieved_at = _now_iso()

        end = date.today()
        start = end - timedelta(days=_DATE_WINDOW_DAYS)
        start_str = start.isoformat()
        end_str = end.isoformat()

        async with httpx.AsyncClient(timeout=30.0) as client:
            biorxiv_resp = await client.get(
                f"{_API_BASE}/details/biorxiv/{start_str}/{end_str}/0"
            )
            biorxiv_resp.raise_for_status()

            medrxiv_resp = await client.get(
                f"{_API_BASE}/details/medrxiv/{start_str}/{end_str}/0"
            )
            medrxiv_resp.raise_for_status()

        parsed = (
            _parse_biorxiv_response(biorxiv_resp.json())
            + _parse_biorxiv_response(medrxiv_resp.json())
        )

        matching = [item for item in parsed if _matches_query(item, keywords)]

        chunks: list[RawChunk] = []
        for item in matching[: self.agent_config.max_results]:
            source_type = MEDRXIV if item["server"] == MEDRXIV else BIORXIV
            chunks.append(
                RawChunk(
                    url=item["url"],
                    text=item["text"],
                    retrieved_at=retrieved_at,
                    source_type=source_type,
                    sub_query=query,
                )
            )

        return chunks
