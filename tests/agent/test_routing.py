"""Tests for mara/agent/edges/routing.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langgraph.types import Send

from mara.agent.edges.routing import route_to_agents
from mara.agents.types import SubQuery


def test_route_to_agents_produces_sends(runnable_config) -> None:
    sq1 = SubQuery(query="q1")
    sq2 = SubQuery(query="q2")
    fake_registry = {"arxiv": object(), "web": object()}
    state = {"sub_queries": [sq1, sq2]}

    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        sends = route_to_agents(state, runnable_config)

    assert len(sends) == 4  # 2 sub-queries × 2 agents
    assert all(isinstance(s, Send) for s in sends)


def test_route_to_agents_send_node_is_run_agent(runnable_config) -> None:
    sq = SubQuery(query="q")
    fake_registry = {"fake": object()}
    state = {"sub_queries": [sq]}

    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        sends = route_to_agents(state, runnable_config)

    assert sends[0].node == "run_agent"


def test_route_to_agents_send_payload(runnable_config) -> None:
    sq = SubQuery(query="test query")
    fake_registry = {"fake": object()}
    state = {"sub_queries": [sq]}

    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        sends = route_to_agents(state, runnable_config)

    payload = sends[0].arg
    assert payload["sub_query"] is sq
    assert payload["agent_type"] == "fake"


def test_route_to_agents_empty_registry(runnable_config) -> None:
    """No agents → returns 'corpus_assembler' string to skip run_agent."""
    state = {"sub_queries": [SubQuery(query="q")]}
    with patch("mara.agent.edges.routing._REGISTRY", {}):
        result = route_to_agents(state, runnable_config)
    assert result == "corpus_assembler"


def test_route_to_agents_empty_sub_queries(runnable_config) -> None:
    """No sub-queries → returns 'corpus_assembler' string."""
    fake_registry = {"arxiv": object()}
    state = {"sub_queries": []}
    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        result = route_to_agents(state, runnable_config)
    assert result == "corpus_assembler"


def test_route_to_agents_missing_sub_queries_key(runnable_config) -> None:
    """Missing sub_queries key should return 'corpus_assembler' without raising."""
    fake_registry = {"arxiv": object()}
    state = {}
    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        result = route_to_agents(state, runnable_config)
    assert result == "corpus_assembler"


# ---------------------------------------------------------------------------
# Directed routing (sub_query.agent is set)
# ---------------------------------------------------------------------------


def test_route_directed_sends_to_named_agent_only(runnable_config) -> None:
    sq = SubQuery(query="q", agent="arxiv")
    fake_registry = {"arxiv": object(), "web": object()}
    state = {"sub_queries": [sq]}

    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        sends = route_to_agents(state, runnable_config)

    assert len(sends) == 1
    assert sends[0].arg["agent_type"] == "arxiv"


def test_route_directed_unknown_agent_broadcasts(runnable_config) -> None:
    """An unrecognized agent name falls back to broadcasting to all agents."""
    sq = SubQuery(query="q", agent="nonexistent")
    fake_registry = {"arxiv": object(), "web": object()}
    state = {"sub_queries": [sq]}

    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        sends = route_to_agents(state, runnable_config)

    assert len(sends) == 2  # broadcast to both
    agent_types = {s.arg["agent_type"] for s in sends}
    assert agent_types == {"arxiv", "web"}


def test_route_mixed_directed_and_broadcast(runnable_config) -> None:
    """Directed sub-queries and broadcast sub-queries can coexist."""
    sq_directed = SubQuery(query="q1", agent="web")
    sq_broadcast = SubQuery(query="q2")  # agent="" → broadcast
    fake_registry = {"arxiv": object(), "web": object()}
    state = {"sub_queries": [sq_directed, sq_broadcast]}

    with patch("mara.agent.edges.routing._REGISTRY", fake_registry):
        sends = route_to_agents(state, runnable_config)

    # directed: 1 send; broadcast: 2 sends
    assert len(sends) == 3
    directed = [s for s in sends if s.arg["sub_query"] is sq_directed]
    broadcast = [s for s in sends if s.arg["sub_query"] is sq_broadcast]
    assert len(directed) == 1
    assert directed[0].arg["agent_type"] == "web"
    assert len(broadcast) == 2
