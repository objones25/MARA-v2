"""LangGraph state definitions for the MARA pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from mara.agents.types import AgentFindings, CertifiedReport, SubQuery, VerifiedChunk
from mara.merkle.forest import ForestTree


class GraphState(TypedDict, total=False):
    """Shared state threaded through the MARA pipeline graph.

    ``total=False`` makes every key optional so nodes can return partial
    updates without providing the full state dict.

    ``findings`` uses ``operator.add`` as a reducer — LangGraph merges
    all ``{"findings": [...]}`` updates from parallel ``run_agent`` nodes
    into a single accumulated list before ``corpus_assembler`` runs.
    """

    original_query: str
    sub_queries: list[SubQuery]
    findings: Annotated[list[AgentFindings], operator.add]
    forest_tree: ForestTree
    flattened_chunks: list[VerifiedChunk]
    selected_chunks: list[VerifiedChunk]
    retrieval_stats: dict[str, int]
    report: str
    certified_report: CertifiedReport


class AgentRunState(TypedDict):
    """Payload sent to each ``run_agent`` node via ``Send()``.

    Contains only the data needed for a single agent × sub-query execution.
    """

    sub_query: SubQuery
    agent_type: str
