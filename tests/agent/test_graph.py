"""Tests for mara/agent/graph.py — build_graph and run_research."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agent.graph import build_graph, run_research
from mara.agents.registry import AgentRegistration
from mara.agents.types import CertifiedReport, SubQuery
from tests.agent.conftest import make_findings


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def test_build_graph_returns_compiled_graph() -> None:
    graph = build_graph()
    # LangGraph compiled graphs expose a .nodes dict
    assert "query_planner" in graph.nodes
    assert "run_agent" in graph.nodes
    assert "corpus_assembler" in graph.nodes
    assert "report_synthesizer" in graph.nodes
    assert "certified_output" in graph.nodes


def test_build_graph_passes_checkpointer_to_compile() -> None:
    mock_checkpointer = MagicMock()
    with patch("mara.agent.graph.StateGraph") as mock_sg:
        mock_builder = MagicMock()
        mock_sg.return_value = mock_builder
        build_graph(checkpointer=mock_checkpointer)
        mock_builder.compile.assert_called_once_with(checkpointer=mock_checkpointer)


def test_build_graph_no_checkpointer_passes_none() -> None:
    with patch("mara.agent.graph.StateGraph") as mock_sg:
        mock_builder = MagicMock()
        mock_sg.return_value = mock_builder
        build_graph()
        mock_builder.compile.assert_called_once_with(checkpointer=None)


# ---------------------------------------------------------------------------
# run_research — end-to-end with all external calls mocked
# ---------------------------------------------------------------------------


def _llm_mock(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


def _fake_agent_cls(agent_type: str):
    """Return a fake agent class that returns a valid AgentFindings."""

    class FakeAgent:
        def __init__(self, config, agent_config):
            self.config = config

        async def run(self, sub_query: SubQuery):
            return make_findings(agent_type=agent_type, query=sub_query.query)

    return FakeAgent


async def test_run_research_returns_certified_report(research_config) -> None:
    planner_response = json.dumps([{"query": "sub-question", "domain": ""}])
    fake_registry = {"fake": AgentRegistration(cls=_fake_agent_cls("fake"))}

    with (
        patch(
            "mara.agent.nodes.query_planner.make_llm",
            return_value=_llm_mock(planner_response),
        ),
        patch(
            "mara.agent.nodes.report_synthesizer.make_llm",
            return_value=_llm_mock("The synthesised report."),
        ),
        patch("mara.agent.nodes.run_agent._REGISTRY", fake_registry),
        patch("mara.agent.edges.routing._REGISTRY", fake_registry),
    ):
        result = await run_research("what is quantum computing?", research_config)

    assert isinstance(result, CertifiedReport)
    assert result.original_query == "what is quantum computing?"
    assert result.report == "The synthesised report."
    assert len(result.chunks) > 0
    assert result.forest_tree.root != ""


async def test_run_research_warns_on_zero_chunk_agent(research_config, capsys) -> None:
    """Agents that return 0 chunks should trigger a stderr warning."""
    planner_response = json.dumps([{"query": "sub-question", "domain": ""}])

    def _zero_chunk_agent_cls(agent_type: str):
        class ZeroChunkAgent:
            def __init__(self, config, agent_config):
                pass

            async def run(self, sub_query: SubQuery):
                return make_findings(agent_type=agent_type, query=sub_query.query, chunks=())

        return ZeroChunkAgent

    fake_registry = {"empty_agent": AgentRegistration(cls=_zero_chunk_agent_cls("empty_agent"))}

    with (
        patch(
            "mara.agent.nodes.query_planner.make_llm",
            return_value=_llm_mock(planner_response),
        ),
        patch(
            "mara.agent.nodes.report_synthesizer.make_llm",
            return_value=_llm_mock("No sources."),
        ),
        patch("mara.agent.nodes.run_agent._REGISTRY", fake_registry),
        patch("mara.agent.edges.routing._REGISTRY", fake_registry),
    ):
        await run_research("what is quantum computing?", research_config)

    assert "empty_agent" in capsys.readouterr().err


async def test_run_research_empty_registry(research_config) -> None:
    """Empty registry → no findings → empty report still yields CertifiedReport."""
    planner_response = json.dumps([{"query": "sub", "domain": ""}])

    with (
        patch(
            "mara.agent.nodes.query_planner.make_llm",
            return_value=_llm_mock(planner_response),
        ),
        patch(
            "mara.agent.nodes.report_synthesizer.make_llm",
            return_value=_llm_mock("Empty corpus report."),
        ),
        patch("mara.agent.nodes.run_agent._REGISTRY", {}),
        patch("mara.agent.edges.routing._REGISTRY", {}),
    ):
        result = await run_research("query", research_config)

    assert isinstance(result, CertifiedReport)
    assert result.chunks == ()
    assert result.report == "Empty corpus report."
