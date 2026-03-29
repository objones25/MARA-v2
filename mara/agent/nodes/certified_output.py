"""Certified output node — assembles the final CertifiedReport."""

from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig

from mara.agent.state import GraphState
from mara.agents.types import CertifiedReport, VerifiedChunk

_log = logging.getLogger(__name__)


def _build_references_section(report: str, chunks: tuple[VerifiedChunk, ...]) -> str:
    """Build a References section from citation indices found in *report*.

    Handles four citation formats the LLM may produce:

    - ``[ML:N:hash]``  — MARA native format
    - ``[N:hash]``     — index+hash without prefix (LLM variant)
    - ``[N]``          — single index shorthand
    - ``[N, M, ...]``  — comma-separated multi-index

    Returns an empty string when no valid citations are found.
    """
    if not chunks:
        return ""

    chunk_by_index: dict[int, VerifiedChunk] = {c.chunk_index: c for c in chunks}
    cited: set[int] = set()

    # [ML:N:hash] and [N:hash] — both capture the numeric index before the hash
    for m in re.finditer(r"\[(?:ML:)?(\d+):[0-9a-f]+\]", report):
        cited.add(int(m.group(1)))

    for m in re.finditer(r"\[(\d+(?:,\s*\d+)*)\]", report):
        for part in m.group(1).split(","):
            cited.add(int(part.strip()))

    valid = sorted(i for i in cited if i in chunk_by_index)
    if not valid:
        return ""

    lines = ["\n## References", ""]
    for idx in valid:
        lines.append(f"[{idx}] {chunk_by_index[idx].url}")
    return "\n".join(lines)


def certified_output_node(state: GraphState, config: RunnableConfig) -> dict:
    """Package the report and provenance into a ``CertifiedReport``.

    Reads ``original_query``, ``report``, ``forest_tree``, and
    ``flattened_chunks`` from ``state``.  Appends a References section
    mapping every cited index to its source URL, then constructs the
    immutable ``CertifiedReport``.
    """
    flattened_chunks = tuple(state.get("flattened_chunks", []))
    report = state["report"]
    references = _build_references_section(report, flattened_chunks)
    if references:
        report = report + "\n" + references

    certified = CertifiedReport(
        original_query=state["original_query"],
        report=report,
        forest_tree=state["forest_tree"],
        chunks=flattened_chunks,
    )
    _log.debug(
        "certified_output: %d chunk(s), forest_root=%s",
        len(certified.chunks),
        certified.forest_tree.root[:8] or "(empty)",
    )
    return {"certified_report": certified}
