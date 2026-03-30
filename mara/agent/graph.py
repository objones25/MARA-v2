"""LangGraph pipeline graph for MARA.

Build order: all node and edge modules must be importable before calling
``build_graph()``.  Import agent modules (e.g. ``mara.agents.arxiv``) before
calling ``run_research()`` so their ``@agent()`` decorators populate
``_REGISTRY``.
"""

from __future__ import annotations

import logging
import sys

from langgraph.graph import END, START, StateGraph

from mara.agent.edges.routing import route_to_agents
from mara.agent.nodes.certified_output import certified_output_node
from mara.agent.nodes.chunk_selector import chunk_selector_node
from mara.agent.nodes.corpus_assembler import corpus_assembler_node
from mara.agent.nodes.query_planner import query_planner_node
from mara.agent.nodes.report_synthesizer import report_synthesizer_node
from mara.agent.nodes.run_agent import run_agent_node
from mara.agent.state import GraphState
from mara.agents.types import CertifiedReport
from mara.config import ResearchConfig

_log = logging.getLogger(__name__)


def build_graph(checkpointer=None) -> StateGraph:
    """Compile and return the MARA research pipeline graph.

    Graph topology::

        START → query_planner → [route_to_agents] → run_agent (×N×M)
              → corpus_assembler → chunk_selector → report_synthesizer
              → certified_output → END

    Fan-out uses LangGraph ``Send()``; fan-in uses the ``operator.add``
    reducer on ``GraphState.findings``.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``SqliteSaver``).
            When provided, the graph can resume interrupted runs via
            ``thread_id`` in the run config.
    """
    builder = StateGraph(GraphState)

    builder.add_node("query_planner", query_planner_node)
    builder.add_node("run_agent", run_agent_node)
    builder.add_node("corpus_assembler", corpus_assembler_node)
    builder.add_node("chunk_selector", chunk_selector_node)
    builder.add_node("report_synthesizer", report_synthesizer_node)
    builder.add_node("certified_output", certified_output_node)

    builder.add_edge(START, "query_planner")
    builder.add_conditional_edges(
        "query_planner", route_to_agents, ["run_agent", "corpus_assembler"]
    )
    builder.add_edge("run_agent", "corpus_assembler")
    builder.add_edge("corpus_assembler", "chunk_selector")
    builder.add_edge("chunk_selector", "report_synthesizer")
    builder.add_edge("report_synthesizer", "certified_output")
    builder.add_edge("certified_output", END)

    return builder.compile(checkpointer=checkpointer)


async def run_research(
    query: str, config: ResearchConfig, thread_id: str | None = None
) -> CertifiedReport:
    """Run the full MARA pipeline for *query* and return a ``CertifiedReport``.

    Args:
        query:     The user's research question.
        config:    Fully-populated ``ResearchConfig`` (all API keys required).
        thread_id: Optional thread identifier for LangGraph checkpointing.
            When provided, pipeline state is persisted to
            ``.mara_checkpoints.db`` and the run can be resumed on failure.
            When ``None`` (default), no checkpointer is used.

    Returns:
        A ``CertifiedReport`` containing the narrative report and its full
        Merkle provenance chain.
    """
    if thread_id is not None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(".mara_checkpoints.db") as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            result = await _invoke(graph, query, config, thread_id)
    else:
        graph = build_graph()
        result = await _invoke(graph, query, config, thread_id=None)

    stats: dict[str, int] = result.get("retrieval_stats", {})
    for agent_type, count in sorted(stats.items()):
        if count == 0:
            print(f"WARNING: {agent_type} returned 0 chunks", file=sys.stderr)
            _log.warning("agent %s returned 0 chunks", agent_type)
    return result["certified_report"]


async def _invoke(graph, query: str, config: ResearchConfig, thread_id: str | None) -> dict:
    """Invoke the compiled graph with the given query and config."""
    configurable: dict = {"research_config": config}
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    return await graph.ainvoke(
        {"original_query": query},
        config={"configurable": configurable},
    )
