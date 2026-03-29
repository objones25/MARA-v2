"""Papers With Code / HuggingFace Papers specialist agent."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
from mara.agents.types import RawChunk, SubQuery

PAPER_ABSTRACT = "paper_abstract"

_HF_BASE = "https://huggingface.co"
_HF_API_BASE = f"{_HF_BASE}/api"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_paper_results(data: list) -> list[dict]:
    """Parse a HuggingFace Papers API response list into url+text dicts."""
    out = []
    for item in data:
        paper_id = (item.get("id") or "").strip()
        if not paper_id:
            continue

        summary = (item.get("summary") or "").strip()
        if not summary:
            continue

        title = (item.get("title") or "").strip()
        text = f"{title}\n{summary}" if title else summary

        out.append({
            "url": f"{_HF_BASE}/papers/{paper_id}",
            "text": text,
        })
    return out


@agent(
    "pwc",
    description="HuggingFace Papers — retrieves ML research paper abstracts with community signals.",
    capabilities=[
        "Search across ML papers indexed on HuggingFace Papers (formerly Papers With Code)",
        "Full paper summaries and abstracts for any ML sub-field",
        "Community upvote signals indicate high-impact or trending work",
        "Covers computer vision, NLP, reinforcement learning, generative models, and more",
    ],
    limitations=[
        "Only covers papers indexed on huggingface.co/papers — not all ML literature",
        "No leaderboard or benchmark metric data; abstracts only",
        "Search relevance depends on HuggingFace indexing, not peer-review status",
        "Poor fit for biomedical, clinical, or non-ML research topics",
    ],
    example_queries=[
        "state-of-the-art vision transformers for image classification",
        "diffusion models for text-to-image generation",
        "reinforcement learning from human feedback for large language models",
        "efficient transformer architectures for low-resource NLP",
    ],
    config=AgentConfig(max_results=20),
)
class PapersWithCodeAgent(SpecialistAgent):
    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        query = sub_query.query
        retrieved_at = _now_iso()

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{_HF_API_BASE}/papers",
                params={"q": query, "limit": self.agent_config.max_results},
            )
            resp.raise_for_status()

        chunks: list[RawChunk] = []
        for item in _parse_paper_results(resp.json()):
            chunks.append(
                RawChunk(
                    url=item["url"],
                    text=item["text"],
                    retrieved_at=retrieved_at,
                    source_type=PAPER_ABSTRACT,
                    sub_query=query,
                )
            )
        return chunks
