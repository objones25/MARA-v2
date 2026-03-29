"""Tests for mara/agent/nodes/chunk_selector.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mara.agent.nodes.chunk_selector import chunk_selector_node
from tests.agent.conftest import make_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    original_query: str = "research query",
    flattened_chunks=None,
) -> dict:
    state: dict = {"original_query": original_query}
    if flattened_chunks is not None:
        state["flattened_chunks"] = flattened_chunks
    return state


# ---------------------------------------------------------------------------
# Empty / missing input
# ---------------------------------------------------------------------------


def test_chunk_selector_empty_chunks(runnable_config) -> None:
    result = chunk_selector_node(_state(flattened_chunks=[]), runnable_config)
    assert result == {"selected_chunks": []}


def test_chunk_selector_missing_flattened_chunks_key(runnable_config) -> None:
    """Missing flattened_chunks key defaults to [] without raising."""
    result = chunk_selector_node(_state(), runnable_config)
    assert result == {"selected_chunks": []}


# ---------------------------------------------------------------------------
# Deduplication by hash
# ---------------------------------------------------------------------------


def test_chunk_selector_deduplicates_by_hash(runnable_config) -> None:
    chunk = make_chunk(text="identical content", chunk_index=0)
    duplicate = make_chunk(text="identical content", chunk_index=1)
    # Both have the same hash because text+url+retrieved_at are equal
    assert chunk.hash == duplicate.hash
    result = chunk_selector_node(
        _state(flattened_chunks=[chunk, duplicate]), runnable_config
    )
    assert len(result["selected_chunks"]) == 1


def test_chunk_selector_keeps_distinct_chunks(runnable_config) -> None:
    chunks = [make_chunk(text=f"distinct text {i}", chunk_index=i) for i in range(5)]
    result = chunk_selector_node(
        _state(flattened_chunks=chunks), runnable_config
    )
    assert len(result["selected_chunks"]) == 5


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------


def test_chunk_selector_applies_cap(runnable_config) -> None:
    # Default cap is 50; give it 60 unique chunks
    chunks = [make_chunk(text=f"unique chunk text number {i}", chunk_index=i) for i in range(60)]
    result = chunk_selector_node(
        _state(flattened_chunks=chunks), runnable_config
    )
    assert len(result["selected_chunks"]) == 50


def test_chunk_selector_below_cap_returns_all(runnable_config) -> None:
    chunks = [make_chunk(text=f"chunk {i}", chunk_index=i) for i in range(10)]
    result = chunk_selector_node(
        _state(flattened_chunks=chunks), runnable_config
    )
    assert len(result["selected_chunks"]) == 10


def test_chunk_selector_custom_cap(runnable_config, research_config) -> None:
    low_cap_config = research_config.model_copy(update={"chunk_selector_cap": 3})
    config = {"configurable": {"research_config": low_cap_config}}

    chunks = [make_chunk(text=f"chunk number {i}", chunk_index=i) for i in range(10)]
    result = chunk_selector_node(_state(flattened_chunks=chunks), config)
    assert len(result["selected_chunks"]) == 3


# ---------------------------------------------------------------------------
# BM25 relevance ordering
# ---------------------------------------------------------------------------


def test_chunk_selector_scores_by_relevance(runnable_config) -> None:
    """Higher-relevance chunk should appear before lower-relevance one."""
    relevant = make_chunk(
        text="transformer attention mechanism self-attention query key value",
        chunk_index=0,
    )
    irrelevant = make_chunk(
        text="medieval tapestry weaving techniques loom shuttle",
        chunk_index=1,
    )
    result = chunk_selector_node(
        _state(
            original_query="transformer attention mechanism",
            flattened_chunks=[irrelevant, relevant],
        ),
        runnable_config,
    )
    selected = result["selected_chunks"]
    assert len(selected) == 2
    assert selected[0].chunk_index == 0  # relevant comes first


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_chunk_selector_returns_dict_with_selected_chunks_key(runnable_config) -> None:
    chunk = make_chunk(chunk_index=0)
    result = chunk_selector_node(_state(flattened_chunks=[chunk]), runnable_config)
    assert "selected_chunks" in result
    assert isinstance(result["selected_chunks"], list)
