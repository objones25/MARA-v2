"""Certified output node — assembles the final CertifiedReport."""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from mara.agent.state import GraphState
from mara.agents.types import CertifiedReport

_log = logging.getLogger(__name__)


def certified_output_node(state: GraphState, config: RunnableConfig) -> dict:
    """Package the report and provenance into a ``CertifiedReport``.

    Reads ``original_query``, ``report``, ``forest_tree``, and
    ``flattened_chunks`` from ``state`` and constructs the immutable
    ``CertifiedReport`` that is the pipeline's final output.
    """
    certified = CertifiedReport(
        original_query=state["original_query"],
        report=state["report"],
        forest_tree=state["forest_tree"],
        chunks=tuple(state.get("flattened_chunks", [])),
    )
    _log.debug(
        "certified_output: %d chunk(s), forest_root=%s",
        len(certified.chunks),
        certified.forest_tree.root[:8] or "(empty)",
    )
    return {"certified_report": certified}
