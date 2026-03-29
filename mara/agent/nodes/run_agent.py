"""Run-agent node — executes a single specialist agent for one sub-query."""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from mara.agent.state import AgentRunState
from mara.agents.registry import _REGISTRY

_log = logging.getLogger(__name__)


async def run_agent_node(state: AgentRunState, config: RunnableConfig) -> dict:
    """Run one registered agent against one sub-query.

    Instantiates the agent class looked up by ``state["agent_type"]`` from
    the module-level ``_REGISTRY`` reference (patchable in tests via
    ``patch("mara.agent.nodes.run_agent._REGISTRY", {...})``).

    Returns ``{"findings": [AgentFindings]}`` so the ``operator.add``
    reducer on ``GraphState.findings`` can accumulate results across all
    parallel invocations.
    """
    research_config = config["configurable"]["research_config"]
    agent_type: str = state["agent_type"]
    sub_query = state["sub_query"]

    _log.debug("run_agent: agent=%r query=%r", agent_type, sub_query.query[:60])

    reg = _REGISTRY[agent_type]
    overrides: dict = getattr(research_config, "agent_config_overrides", {})
    agent_config = overrides.get(agent_type, reg.config)
    agent = reg.cls(research_config, agent_config)
    try:
        findings = await agent.run(sub_query)
    except Exception as exc:
        _log.warning(
            "run_agent: agent=%r failed for query=%r: %s",
            agent_type,
            sub_query.query[:60],
            exc,
        )
        return {"findings": []}

    _log.debug(
        "run_agent: agent=%r returned %d chunk(s)", agent_type, findings.chunk_count
    )
    return {"findings": [findings]}
