"""Tests for mara/agent/nodes/certified_output.py."""

from __future__ import annotations

import pytest

from mara.agent.nodes.certified_output import certified_output_node
from mara.agents.types import CertifiedReport
from mara.merkle.forest import ForestTree
from tests.agent.conftest import make_chunk, make_findings


def _make_forest() -> ForestTree:
    from mara.merkle.forest import build_forest_tree
    findings = make_findings(agent_type="arxiv")
    return build_forest_tree([("arxiv", findings.merkle_root)], "sha256")


def test_certified_output_returns_certified_report(runnable_config) -> None:
    chunk = make_chunk(chunk_index=0)
    forest = _make_forest()
    state = {
        "original_query": "what is AI?",
        "report": "AI is ...",
        "forest_tree": forest,
        "flattened_chunks": [chunk],
    }

    result = certified_output_node(state, runnable_config)

    cr = result["certified_report"]
    assert isinstance(cr, CertifiedReport)
    assert cr.original_query == "what is AI?"
    assert cr.report == "AI is ..."
    assert cr.forest_tree is forest
    assert cr.chunks == (chunk,)


def test_certified_output_empty_chunks(runnable_config) -> None:
    forest = ForestTree(algorithm="sha256")
    state = {
        "original_query": "q",
        "report": "r",
        "forest_tree": forest,
        "flattened_chunks": [],
    }

    result = certified_output_node(state, runnable_config)

    assert result["certified_report"].chunks == ()


def test_certified_output_missing_flattened_chunks(runnable_config) -> None:
    """Missing flattened_chunks defaults to empty tuple without raising."""
    forest = ForestTree(algorithm="sha256")
    state = {
        "original_query": "q",
        "report": "r",
        "forest_tree": forest,
    }

    result = certified_output_node(state, runnable_config)

    assert result["certified_report"].chunks == ()


def test_certified_output_report_is_immutable(runnable_config) -> None:
    forest = ForestTree(algorithm="sha256")
    state = {
        "original_query": "q",
        "report": "r",
        "forest_tree": forest,
        "flattened_chunks": [],
    }
    result = certified_output_node(state, runnable_config)
    cr = result["certified_report"]
    # CertifiedReport is frozen — mutation must raise
    with pytest.raises((AttributeError, TypeError)):
        cr.report = "mutated"  # type: ignore[misc]
