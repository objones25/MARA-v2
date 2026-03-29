"""LangGraph pipeline graph for MARA.

Build order: all node and edge modules must be importable before calling
``build_graph()``.  Import agent modules (e.g. ``mara.agents.arxiv``) before
calling ``run_research()`` so their ``@agent()`` decorators populate
``_REGISTRY``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from mara.agent.edges.routing import route_to_agents
from mara.agent.nodes.certified_output import certified_output_node
from mara.agent.nodes.corpus_assembler import corpus_assembler_node
from mara.agent.nodes.query_planner import query_planner_node
from mara.agent.nodes.report_synthesizer import report_synthesizer_node
from mara.agent.nodes.run_agent import run_agent_node
from mara.agent.state import GraphState
from mara.agents.types import CertifiedReport
from mara.config import ResearchConfig


def build_graph() -> StateGraph:
    """Compile and return the MARA research pipeline graph.

    Graph topology::

        START → query_planner → [route_to_agents] → run_agent (×N×M)
              → corpus_assembler → report_synthesizer → certified_output → END

    Fan-out uses LangGraph ``Send()``; fan-in uses the ``operator.add``
    reducer on ``GraphState.findings``.
    """
    builder = StateGraph(GraphState)

    builder.add_node("query_planner", query_planner_node)
    builder.add_node("run_agent", run_agent_node)
    builder.add_node("corpus_assembler", corpus_assembler_node)
    builder.add_node("report_synthesizer", report_synthesizer_node)
    builder.add_node("certified_output", certified_output_node)

    builder.add_edge(START, "query_planner")
    builder.add_conditional_edges(
        "query_planner", route_to_agents, ["run_agent", "corpus_assembler"]
    )
    builder.add_edge("run_agent", "corpus_assembler")
    builder.add_edge("corpus_assembler", "report_synthesizer")
    builder.add_edge("report_synthesizer", "certified_output")
    builder.add_edge("certified_output", END)

    return builder.compile()


async def run_research(query: str, config: ResearchConfig) -> CertifiedReport:
    """Run the full MARA pipeline for *query* and return a ``CertifiedReport``.

    Args:
        query:  The user's research question.
        config: Fully-populated ``ResearchConfig`` (all API keys required).

    Returns:
        A ``CertifiedReport`` containing the narrative report and its full
        Merkle provenance chain.
    """
    graph = build_graph()
    result = await graph.ainvoke(
        {"original_query": query},
        config={"configurable": {"research_config": config}},
    )
    return result["certified_report"]
