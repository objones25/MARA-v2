"""Corpus assembler node — merges all agent findings into a single provenance structure."""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict

from langchain_core.runnables import RunnableConfig

from mara.agent.state import GraphState
from mara.agents.types import AgentFindings, VerifiedChunk
from mara.merkle.forest import build_forest_tree
from mara.merkle.tree import build_merkle_tree

_log = logging.getLogger(__name__)


def corpus_assembler_node(state: GraphState, config: RunnableConfig) -> dict:
    """Merge all ``AgentFindings`` into a ``ForestTree`` and a flat chunk list.

    Multiple ``AgentFindings`` objects may share the same ``agent_type``
    (one per sub-query × agent). This node:

    1. Groups chunks by ``agent_type``.
    2. Sorts each group by ``(sub_query, chunk_index)`` for determinism.
    3. Builds one ``MerkleTree`` per agent from the merged chunk list.
    4. Calls ``build_forest_tree`` with one ``(agent_type, root)`` pair per
       agent (satisfying its uniqueness requirement).
    5. Assigns globally unique ``chunk_index`` values across all agents
       using ``dataclasses.replace`` (``VerifiedChunk`` is frozen).
    """
    research_config = config["configurable"]["research_config"]
    algorithm = research_config.hash_algorithm
    findings_list: list[AgentFindings] = state.get("findings", [])

    # Group chunks by agent_type
    agent_chunks: dict[str, list[VerifiedChunk]] = defaultdict(list)
    for findings in findings_list:
        agent_chunks[findings.agent_type].extend(findings.chunks)

    # Sort each agent's chunks deterministically
    for chunks in agent_chunks.values():
        chunks.sort(key=lambda c: (c.sub_query, c.chunk_index))

    # Build one MerkleTree root per agent and collect for ForestTree.
    # Skip agents that returned no chunks — their empty root ("") is not a
    # valid leaf hash and build_forest_tree would raise ValueError.
    agent_data: list[tuple[str, str]] = []
    for agent_type in sorted(agent_chunks):
        chunks = agent_chunks[agent_type]
        if not chunks:
            continue
        tree = build_merkle_tree([c.hash for c in chunks], algorithm)
        agent_data.append((agent_type, tree.root))

    forest_tree = build_forest_tree(agent_data, algorithm)

    # Flatten all chunks in sorted-agent order and assign global indices
    all_chunks: list[VerifiedChunk] = []
    for agent_type in sorted(agent_chunks):
        all_chunks.extend(agent_chunks[agent_type])

    flattened = [
        dataclasses.replace(c, chunk_index=i) for i, c in enumerate(all_chunks)
    ]

    retrieval_stats = {
        agent_type: len(chunks) for agent_type, chunks in agent_chunks.items()
    }

    _log.debug(
        "corpus_assembler: %d agent(s), %d total chunk(s), forest_root=%s",
        len(agent_data),
        len(flattened),
        forest_tree.root[:8] or "(empty)",
    )
    return {
        "forest_tree": forest_tree,
        "flattened_chunks": flattened,
        "retrieval_stats": retrieval_stats,
    }
