"""PubMed specialist agent.

Discovers papers via NCBI eUtils (esearch + esummary) and retrieves content
via efetch.  Two content strategies:

1. PMC full text — parses ``<sec>`` XML sections (``source_type="pmc_xml"``)
2. Abstract only — extracts ``<AbstractText>`` (``source_type="abstract_only"``)

Rate limiting (3 req/s without key, 10 req/s with key) is enforced by the
``SpecialistAgent`` base class via ``_get_rate_limit_interval()``.  Individual
``asyncio.sleep`` calls within ``_search()`` provide intra-invocation pacing
between the sequential eUtils requests for each paper.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.pubmed.fulltext import parse_abstract_xml, parse_pmc_sections
from mara.agents.pubmed.search import parse_article_metadata
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

_log = logging.getLogger(__name__)

PMC_XML = "pmc_xml"
ABSTRACT_ONLY = "abstract_only"

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PUBMED_PAPER_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@agent(
    "pubmed",
    description="Retrieves biomedical and clinical research papers from PubMed/PMC via NCBI eUtils.",
    capabilities=[
        "Full-text PMC articles when available; abstract fallback otherwise",
        "Authoritative source for clinical trials, medical case studies, and pharmacology",
        "Strong coverage of life sciences: genomics, proteomics, neuroscience, epidemiology",
        "Peer-reviewed content from established journals",
    ],
    limitations=[
        "Limited to biomedical/life sciences — poor fit for CS, physics, or social sciences",
        "Full text only available for open-access PMC articles",
        "Rate-limited — avoid routing many sub-queries here simultaneously",
        "Weak on very recent preprints (use ArXiv or S2 for cutting-edge CS/physics)",
    ],
    example_queries=[
        "CRISPR-Cas9 off-target effects in vivo",
        "efficacy of mRNA vaccines against SARS-CoV-2 variants",
        "neural correlates of working memory in fMRI studies",
        "antibiotic resistance mechanisms in gram-negative bacteria",
    ],
    config=AgentConfig(rate_limit_rps=3.0),
)
class PubMedAgent(SpecialistAgent):
    """Retrieves research papers from PubMed / PMC via NCBI eUtils."""

    def _ncbi_params(self, **kwargs: object) -> dict:
        """Build an NCBI params dict, including ``api_key`` only when configured."""
        params: dict = dict(kwargs)
        if self.agent_config.api_key:
            params["api_key"] = self.agent_config.api_key
        return params

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch papers from PubMed for *sub_query*.

        Orchestrates three eUtils calls per paper:
        1. esearch — get PMID list
        2. esummary — get metadata (title, PMC ID if available)
        3. efetch — get full text (PMC XML) or abstract

        Raises:
            httpx.HTTPStatusError: if esearch returns a non-2xx response.
        """
        retrieved_at = _now_iso()
        delay = self._get_rate_limit_interval()

        async with httpx.AsyncClient() as client:
            # 1. esearch — discover PMIDs
            esearch_resp = await client.get(
                _ESEARCH_URL,
                params=self._ncbi_params(
                    db="pubmed",
                    term=sub_query.query,
                    retmax=self.agent_config.max_results,
                    retmode="json",
                ),
            )
            await asyncio.sleep(delay)

            esearch_resp.raise_for_status()
            pmids: list[str] = esearch_resp.json().get("esearchresult", {}).get("idlist", [])
            _log.debug("pubmed esearch: %d PMID(s) for %r", len(pmids), sub_query.query[:60])

            if not pmids:
                return []

            # 2. esummary — fetch metadata for all PMIDs in one request
            esummary_resp = await client.get(
                _ESUMMARY_URL,
                params=self._ncbi_params(
                    db="pubmed",
                    id=",".join(pmids),
                    retmode="json",
                ),
            )
            await asyncio.sleep(delay)

            esummary_resp.raise_for_status()
            summary_result = esummary_resp.json().get("result", {})

            # 3. Fetch content for each article
            chunks: list[RawChunk] = []
            for pmid in pmids:
                meta = parse_article_metadata(summary_result.get(pmid, {}))
                if meta is None:
                    _log.debug("pubmed: skipping PMID %s (no usable metadata)", pmid)
                    continue

                url = _PUBMED_PAPER_URL.format(pmid=pmid)

                if meta["pmc_id"]:
                    # PMC full text
                    efetch_resp = await client.get(
                        _EFETCH_URL,
                        params=self._ncbi_params(
                            db="pmc",
                            id=meta["pmc_id"],
                            rettype="full",
                            retmode="xml",
                        ),
                    )
                    await asyncio.sleep(delay)

                    efetch_resp.raise_for_status()
                    sections = parse_pmc_sections(efetch_resp.text)
                    for section_text in sections:
                        chunks.append(
                            RawChunk(
                                url=url,
                                text=section_text,
                                retrieved_at=retrieved_at,
                                source_type=PMC_XML,
                                sub_query=sub_query.query,
                            )
                        )
                    _log.debug(
                        "pubmed pmc: PMID %s → %d section(s)", pmid, len(sections)
                    )
                else:
                    # Abstract fallback
                    efetch_resp = await client.get(
                        _EFETCH_URL,
                        params=self._ncbi_params(
                            db="pubmed",
                            id=pmid,
                            rettype="abstract",
                            retmode="xml",
                        ),
                    )
                    await asyncio.sleep(delay)

                    efetch_resp.raise_for_status()
                    abstract = parse_abstract_xml(efetch_resp.text)
                    if abstract:
                        chunks.append(
                            RawChunk(
                                url=url,
                                text=abstract,
                                retrieved_at=retrieved_at,
                                source_type=ABSTRACT_ONLY,
                                sub_query=sub_query.query,
                            )
                        )
                    _log.debug(
                        "pubmed abstract: PMID %s → %d char(s)", pmid, len(abstract)
                    )

        return chunks
