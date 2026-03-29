"""Routing function that fans out sub-queries to registered agents."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from mara.agent.state import GraphState
from mara.agents.registry import _REGISTRY


def route_to_agents(
    state: GraphState, config: RunnableConfig
) -> list[Send] | str:
    """Return ``Send("run_agent", ...)`` objects for each sub-query.

    Routing strategy (per sub-query):
    - **Directed**: if ``sub_query.agent`` names a registered agent, send to
      that agent only.
    - **Broadcast fallback**: if ``sub_query.agent`` is empty or unrecognized,
      send to every registered agent (preserves original behavior).

    Called as a conditional-edge function from ``query_planner``.  Returns
    ``"corpus_assembler"`` (a plain string edge) when either the sub-query
    list or the agent registry is empty.
    """
    sub_queries = state.get("sub_queries", [])
    agent_types = list(_REGISTRY.keys())

    sends: list[Send] = []
    for sq in sub_queries:
        if sq.agent and sq.agent in _REGISTRY:
            sends.append(Send("run_agent", {"sub_query": sq, "agent_type": sq.agent}))
        else:
            for at in agent_types:
                sends.append(Send("run_agent", {"sub_query": sq, "agent_type": at}))

    return sends if sends else "corpus_assembler"
