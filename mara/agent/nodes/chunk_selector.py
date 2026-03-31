"""Chunk selector node — deduplicates and ranks chunks before synthesis."""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import zip_longest

from langchain_core.runnables import RunnableConfig

from mara.agent.scoring import score_chunks_bm25
from mara.agent.state import GraphState
from mara.agents.types import VerifiedChunk

_log = logging.getLogger(__name__)


def _interleave_by_sub_query(
    scored: list[tuple[VerifiedChunk, float]],
) -> list[tuple[VerifiedChunk, float]]:
    """Round-robin interleave BM25-ranked chunks by sub_query.

    Groups chunks by their originating ``sub_query`` string, preserving BM25
    rank within each group.  The round-robin then picks one chunk per group per
    round, ensuring every research aspect gets representation in the cap window
    before any single sub_query can exhaust it.

    When all chunks share the same sub_query (single-aspect queries or tests)
    the function is a no-op and the original BM25 order is returned unchanged.
    """
    groups: dict[str, list[tuple[VerifiedChunk, float]]] = defaultdict(list)
    for item in scored:
        groups[item[0].sub_query].append(item)
    result: list[tuple[VerifiedChunk, float]] = []
    for round_ in zip_longest(*groups.values()):
        for item in round_:
            if item is not None:
                result.append(item)
    return result


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

    # Step 2b: round-robin interleave by sub_query for topical diversity
    scored = _interleave_by_sub_query(scored)

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
