"""Tests for mara/agent/nodes/certified_output.py."""

from __future__ import annotations

import pytest

from mara.agent.nodes.certified_output import _build_references_section, certified_output_node
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


# ---------------------------------------------------------------------------
# _build_references_section
# ---------------------------------------------------------------------------


def test_build_references_no_citations() -> None:
    chunk = make_chunk(chunk_index=0)
    result = _build_references_section("No citations here.", (chunk,))
    assert result == ""


def test_build_references_empty_chunks() -> None:
    result = _build_references_section("Text with [0] citation.", ())
    assert result == ""


def test_build_references_single_index() -> None:
    chunk = make_chunk(url="https://example.com/paper", chunk_index=3)
    result = _build_references_section("See [3] for details.", (chunk,))
    assert "[3] https://example.com/paper" in result
    assert "## References" in result


def test_build_references_multi_index() -> None:
    chunks = (
        make_chunk(url="https://example.com/a", chunk_index=1),
        make_chunk(url="https://example.com/b", chunk_index=5),
    )
    result = _build_references_section("As shown in [1, 5].", chunks)
    assert "[1] https://example.com/a" in result
    assert "[5] https://example.com/b" in result


def test_build_references_ml_format() -> None:
    chunk = make_chunk(url="https://arxiv.org/abs/1234", chunk_index=2)
    result = _build_references_section("Cited [ML:2:abcd1234].", (chunk,))
    assert "[2] https://arxiv.org/abs/1234" in result


def test_build_references_index_hash_format() -> None:
    """[N:hash] variant (LLM omits the ML: prefix) must still resolve."""
    chunk = make_chunk(url="https://arxiv.org/abs/5678", chunk_index=7)
    result = _build_references_section("Cited [7:abcd1234].", (chunk,))
    assert "[7] https://arxiv.org/abs/5678" in result


def test_build_references_out_of_range_index_skipped() -> None:
    chunk = make_chunk(chunk_index=0)
    result = _build_references_section("See [99] which is out of range.", (chunk,))
    assert result == ""


def test_build_references_indices_sorted() -> None:
    chunks = (
        make_chunk(url="https://example.com/z", chunk_index=5),
        make_chunk(url="https://example.com/a", chunk_index=1),
    )
    result = _build_references_section("[5] and [1] cited.", chunks)
    lines = [l for l in result.splitlines() if l.startswith("[")]
    assert lines[0].startswith("[1]")
    assert lines[1].startswith("[5]")


def test_build_references_deduplicates_same_index() -> None:
    chunk = make_chunk(url="https://example.com/paper", chunk_index=0)
    result = _build_references_section("[0] first mention [0] second mention.", (chunk,))
    assert result.count("[0]") == 1


# ---------------------------------------------------------------------------
# certified_output_node — references integration
# ---------------------------------------------------------------------------


def test_certified_output_appends_references_when_citations_present(runnable_config) -> None:
    chunk = make_chunk(url="https://example.com/source", chunk_index=0)
    forest = _make_forest()
    state = {
        "original_query": "what is AI?",
        "report": "AI is interesting [0].",
        "forest_tree": forest,
        "flattened_chunks": [chunk],
    }

    result = certified_output_node(state, runnable_config)
    report = result["certified_report"].report

    assert "## References" in report
    assert "[0] https://example.com/source" in report


def test_certified_output_no_references_when_no_citations(runnable_config) -> None:
    chunk = make_chunk(chunk_index=0)
    forest = _make_forest()
    state = {
        "original_query": "q",
        "report": "No citations in this report.",
        "forest_tree": forest,
        "flattened_chunks": [chunk],
    }

    result = certified_output_node(state, runnable_config)
    assert "## References" not in result["certified_report"].report


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
