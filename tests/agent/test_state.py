"""Tests for mara/agent/state.py."""

from __future__ import annotations

import operator
import typing

import pytest

from mara.agent.state import AgentRunState, GraphState
from mara.agents.types import SubQuery


def test_graph_state_is_total_false():
    """GraphState should allow partial dicts (total=False)."""
    # A TypedDict with total=False means all keys are optional
    state: GraphState = {"original_query": "hello"}
    assert state["original_query"] == "hello"


def test_graph_state_findings_has_add_reducer():
    """GraphState.findings must be annotated with operator.add."""
    # Use include_extras=True so Annotated metadata is preserved even with
    # 'from __future__ import annotations' (which stringifies annotations).
    hints = typing.get_type_hints(GraphState, include_extras=True)
    assert "findings" in hints
    metadata = getattr(hints["findings"], "__metadata__", ())
    assert operator.add in metadata


def test_agent_run_state_keys():
    """AgentRunState must have sub_query and agent_type keys."""
    state: AgentRunState = {
        "sub_query": SubQuery(query="test"),
        "agent_type": "arxiv",
    }
    assert state["agent_type"] == "arxiv"
    assert state["sub_query"].query == "test"
