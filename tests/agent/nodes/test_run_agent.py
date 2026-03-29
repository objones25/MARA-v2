"""Tests for mara/agent/nodes/run_agent.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agent.nodes.run_agent import run_agent_node
from mara.agents.types import SubQuery
from tests.agent.conftest import make_findings


def _fake_agent_cls(findings_to_return):
    """Return a fake agent class whose run() returns findings_to_return."""

    class FakeAgent:
        def __init__(self, config):
            self.config = config

        async def run(self, sub_query):
            return findings_to_return

    return FakeAgent


async def test_run_agent_node_returns_findings_list(runnable_config) -> None:
    findings = make_findings(agent_type="fake", query="q")
    fake_registry = {"fake": _fake_agent_cls(findings)}

    state = {"sub_query": SubQuery(query="q"), "agent_type": "fake"}
    with patch("mara.agent.nodes.run_agent._REGISTRY", fake_registry):
        result = await run_agent_node(state, runnable_config)

    assert result == {"findings": [findings]}


async def test_run_agent_node_instantiates_with_config(runnable_config) -> None:
    findings = make_findings(agent_type="fake", query="q")
    received_config = {}

    class FakeAgent:
        def __init__(self, config):
            received_config["config"] = config

        async def run(self, sub_query):
            return findings

    fake_registry = {"fake": FakeAgent}
    state = {"sub_query": SubQuery(query="q"), "agent_type": "fake"}

    with patch("mara.agent.nodes.run_agent._REGISTRY", fake_registry):
        await run_agent_node(state, runnable_config)

    assert received_config["config"] is runnable_config["configurable"]["research_config"]


async def test_run_agent_node_key_error_on_unknown_agent(runnable_config) -> None:
    state = {"sub_query": SubQuery(query="q"), "agent_type": "nonexistent"}
    with patch("mara.agent.nodes.run_agent._REGISTRY", {}):
        with pytest.raises(KeyError):
            await run_agent_node(state, runnable_config)


async def test_run_agent_node_returns_empty_on_agent_exception(runnable_config) -> None:
    class FailingAgent:
        def __init__(self, config):
            pass

        async def run(self, sub_query):
            raise RuntimeError("boom")

    fake_registry = {"failing": FailingAgent}
    state = {"sub_query": SubQuery(query="q"), "agent_type": "failing"}

    with patch("mara.agent.nodes.run_agent._REGISTRY", fake_registry):
        result = await run_agent_node(state, runnable_config)

    assert result == {"findings": []}
