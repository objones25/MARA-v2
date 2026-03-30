"""Tests for mara/agent/nodes/corpus_assembler.py."""

from __future__ import annotations

import pytest

from mara.agent.nodes.corpus_assembler import corpus_assembler_node
from mara.agents.types import SubQuery
from mara.merkle.forest import ForestTree
from tests.agent.conftest import make_chunk, make_findings


# ---------------------------------------------------------------------------
# Empty findings
# ---------------------------------------------------------------------------


def test_corpus_assembler_empty_findings(runnable_config) -> None:
    result = corpus_assembler_node({"original_query": "q", "findings": []}, runnable_config)
    assert result["flattened_chunks"] == []
    assert isinstance(result["forest_tree"], ForestTree)
    assert result["forest_tree"].root == ""


def test_corpus_assembler_missing_findings_key(runnable_config) -> None:
    """State without 'findings' key should not raise — defaults to []."""
    result = corpus_assembler_node({"original_query": "q"}, runnable_config)
    assert result["flattened_chunks"] == []


# ---------------------------------------------------------------------------
# Single agent, single sub-query
# ---------------------------------------------------------------------------


def test_corpus_assembler_single_agent_single_findings(runnable_config) -> None:
    c0 = make_chunk(chunk_index=0, sub_query="q1")
    c1 = make_chunk(url="https://b.com", text="other text", chunk_index=1, sub_query="q1")
    findings = make_findings(agent_type="arxiv", query="q1", chunks=(c0, c1))

    result = corpus_assembler_node({"findings": [findings]}, runnable_config)

    flat = result["flattened_chunks"]
    assert len(flat) == 2
    # Global indices must be reassigned from 0
    assert flat[0].chunk_index == 0
    assert flat[1].chunk_index == 1


def test_corpus_assembler_assigns_global_indices(runnable_config) -> None:
    c0 = make_chunk(chunk_index=5, sub_query="q")  # agent-local index 5
    findings = make_findings(agent_type="web", query="q", chunks=(c0,))

    result = corpus_assembler_node({"findings": [findings]}, runnable_config)

    assert result["flattened_chunks"][0].chunk_index == 0  # reset to global 0


# ---------------------------------------------------------------------------
# Multiple agents
# ---------------------------------------------------------------------------


def test_corpus_assembler_multiple_agents(runnable_config) -> None:
    ca = make_chunk(url="https://a.com", text="arxiv text", sub_query="q", chunk_index=0)
    cw = make_chunk(url="https://w.com", text="web text", sub_query="q", chunk_index=0)
    f_arxiv = make_findings(agent_type="arxiv", query="q", chunks=(ca,))
    f_web = make_findings(agent_type="web", query="q", chunks=(cw,))

    result = corpus_assembler_node({"findings": [f_arxiv, f_web]}, runnable_config)

    flat = result["flattened_chunks"]
    assert len(flat) == 2
    # Sorted alphabetically by agent_type: arxiv first, web second
    assert flat[0].url == "https://a.com"
    assert flat[1].url == "https://w.com"
    assert flat[0].chunk_index == 0
    assert flat[1].chunk_index == 1


def test_corpus_assembler_forest_tree_has_all_agents(runnable_config) -> None:
    ca = make_chunk(url="https://a.com", text="a", sub_query="q")
    cw = make_chunk(url="https://w.com", text="w", sub_query="q")
    f_arxiv = make_findings(agent_type="arxiv", query="q", chunks=(ca,))
    f_web = make_findings(agent_type="web", query="q", chunks=(cw,))

    result = corpus_assembler_node({"findings": [f_arxiv, f_web]}, runnable_config)

    forest = result["forest_tree"]
    assert set(forest.agent_roots.keys()) == {"arxiv", "web"}
    assert forest.root != ""


# ---------------------------------------------------------------------------
# Multiple sub-queries for the same agent (merge logic)
# ---------------------------------------------------------------------------


def test_corpus_assembler_merges_same_agent_across_sub_queries(runnable_config) -> None:
    c0 = make_chunk(url="https://a.com", text="chunk from q1", sub_query="q1", chunk_index=0)
    c1 = make_chunk(url="https://b.com", text="chunk from q2", sub_query="q2", chunk_index=0)
    f1 = make_findings(agent_type="arxiv", query="q1", chunks=(c0,))
    f2 = make_findings(agent_type="arxiv", query="q2", chunks=(c1,))

    result = corpus_assembler_node({"findings": [f1, f2]}, runnable_config)

    flat = result["flattened_chunks"]
    assert len(flat) == 2
    # forest_tree has exactly one entry for arxiv
    forest = result["forest_tree"]
    assert list(forest.agent_roots.keys()) == ["arxiv"]


def test_corpus_assembler_skips_agent_with_zero_chunks(runnable_config) -> None:
    """An AgentFindings with no chunks must not contribute a leaf to the forest tree."""
    c = make_chunk(url="https://a.com", text="arxiv text", sub_query="q", chunk_index=0)
    f_arxiv = make_findings(agent_type="arxiv", query="q", chunks=(c,))
    f_pubmed = make_findings(agent_type="pubmed", query="q", chunks=())  # zero chunks

    result = corpus_assembler_node({"findings": [f_arxiv, f_pubmed]}, runnable_config)

    forest = result["forest_tree"]
    assert "arxiv" in forest.agent_roots
    assert "pubmed" not in forest.agent_roots
    assert forest.root != ""


def test_corpus_assembler_all_agents_zero_chunks(runnable_config) -> None:
    """When every agent returns 0 chunks the forest tree root must be empty string."""
    f = make_findings(agent_type="pubmed", query="q", chunks=())

    result = corpus_assembler_node({"findings": [f]}, runnable_config)

    assert result["flattened_chunks"] == []
    assert result["forest_tree"].root == ""


# ---------------------------------------------------------------------------
# retrieval_stats
# ---------------------------------------------------------------------------


def test_corpus_assembler_retrieval_stats_single_agent(runnable_config) -> None:
    c0 = make_chunk(chunk_index=0, sub_query="q")
    c1 = make_chunk(url="https://b.com", text="b", chunk_index=1, sub_query="q")
    findings = make_findings(agent_type="arxiv", query="q", chunks=(c0, c1))

    result = corpus_assembler_node({"findings": [findings]}, runnable_config)

    assert result["retrieval_stats"] == {"arxiv": 2}


def test_corpus_assembler_retrieval_stats_multiple_agents(runnable_config) -> None:
    ca = make_chunk(url="https://a.com", text="a", sub_query="q", chunk_index=0)
    cw = make_chunk(url="https://w.com", text="w", sub_query="q", chunk_index=0)
    f_arxiv = make_findings(agent_type="arxiv", query="q", chunks=(ca,))
    f_web = make_findings(agent_type="web", query="q", chunks=(cw,))

    result = corpus_assembler_node({"findings": [f_arxiv, f_web]}, runnable_config)

    assert result["retrieval_stats"] == {"arxiv": 1, "web": 1}


def test_corpus_assembler_retrieval_stats_zero_chunk_agent(runnable_config) -> None:
    c = make_chunk(url="https://a.com", text="a", sub_query="q", chunk_index=0)
    f_arxiv = make_findings(agent_type="arxiv", query="q", chunks=(c,))
    f_pubmed = make_findings(agent_type="pubmed", query="q", chunks=())

    result = corpus_assembler_node({"findings": [f_arxiv, f_pubmed]}, runnable_config)

    assert result["retrieval_stats"]["arxiv"] == 1
    assert result["retrieval_stats"]["pubmed"] == 0


def test_corpus_assembler_retrieval_stats_empty_findings(runnable_config) -> None:
    result = corpus_assembler_node({"original_query": "q", "findings": []}, runnable_config)

    assert result["retrieval_stats"] == {}


def test_corpus_assembler_chunk_sort_order(runnable_config) -> None:
    """Chunks within an agent are sorted by (sub_query, chunk_index)."""
    c_q2_0 = make_chunk(url="https://a.com", text="q2 first", sub_query="q2", chunk_index=0)
    c_q1_0 = make_chunk(url="https://b.com", text="q1 first", sub_query="q1", chunk_index=0)
    f1 = make_findings(agent_type="arxiv", query="q2", chunks=(c_q2_0,))
    f2 = make_findings(agent_type="arxiv", query="q1", chunks=(c_q1_0,))

    result = corpus_assembler_node({"findings": [f1, f2]}, runnable_config)

    flat = result["flattened_chunks"]
    # q1 sorts before q2
    assert flat[0].sub_query == "q1"
    assert flat[1].sub_query == "q2"
