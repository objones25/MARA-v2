"""Routing function that fans out sub-queries to all registered agents."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from mara.agent.state import GraphState
from mara.agents.registry import _REGISTRY


def route_to_agents(
    state: GraphState, config: RunnableConfig
) -> list[Send] | str:
    """Return one ``Send("run_agent", ...)`` per sub-query × agent-type pair.

    Called as a conditional-edge function from ``query_planner``.  The
    returned ``Send`` objects fan out execution to all registered specialist
    agents in parallel, each receiving its own ``AgentRunState`` payload.

    Returns ``"corpus_assembler"`` (a plain string edge) when either the
    sub-query list or the agent registry is empty so the graph bypasses
    ``run_agent`` and proceeds directly to assembly with no findings.
    """
    sub_queries = state.get("sub_queries", [])
    agent_types = list(_REGISTRY.keys())
    sends = [
        Send("run_agent", {"sub_query": sq, "agent_type": at})
        for sq in sub_queries
        for at in agent_types
    ]
    return sends if sends else "corpus_assembler"
