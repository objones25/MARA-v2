"""Chunk selector node — deduplicates and ranks chunks before synthesis."""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from mara.agent.scoring import score_chunks_bm25
from mara.agent.state import GraphState
from mara.agents.types import VerifiedChunk

_log = logging.getLogger(__name__)


def chunk_selector_node(state: GraphState, config: RunnableConfig) -> dict:
    """Deduplicate, score, and cap chunks from the assembled corpus.

    Reads ``flattened_chunks`` and ``original_query`` from *state*.

    Steps:
    1. Deduplicate by chunk hash (first occurrence wins).
    2. Score remaining chunks by BM25 keyword overlap against the query.
    3. Truncate to ``research_config.chunk_selector_cap`` (default 50).

    Returns ``{"selected_chunks": [...]}`` for downstream nodes.
    """
    research_config = config["configurable"]["research_config"]
    original_query: str = state.get("original_query", "")
    raw: list[VerifiedChunk] = state.get("flattened_chunks", [])

    # Step 1: deduplicate by hash preserving first occurrence
    seen: set[str] = set()
    deduped: list[VerifiedChunk] = []
    for chunk in raw:
        if chunk.hash not in seen:
            seen.add(chunk.hash)
            deduped.append(chunk)

    if not deduped:
        return {"selected_chunks": []}

    # Step 2: BM25 relevance scoring
    scored = score_chunks_bm25(deduped, original_query, research_config.hash_algorithm)

    # Step 3: cap
    cap = research_config.chunk_selector_cap
    selected = [chunk for chunk, _ in scored[:cap]]

    _log.debug(
        "chunk_selector: %d in → %d deduped → %d selected",
        len(raw),
        len(deduped),
        len(selected),
    )

    return {"selected_chunks": selected}
