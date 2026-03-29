"""Query planner node — decomposes the user query into focused sub-queries."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from mara.agent.state import GraphState
from mara.agents.types import SubQuery
from mara.llm import make_llm

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research query decomposition specialist. "
    "Given a research query, decompose it into 3-5 focused sub-queries that "
    "together cover the topic comprehensively. "
    "Return a JSON array of objects with \"query\" and \"domain\" fields. "
    "Example: [{\"query\": \"...\", \"domain\": \"empirical\"}, ...]. "
    "Return ONLY the JSON array, no other text."
)


async def query_planner_node(state: GraphState, config: RunnableConfig) -> dict:
    """Decompose ``original_query`` into a list of ``SubQuery`` objects.

    Calls the LLM with a decomposition prompt and parses the JSON response.
    Falls back to a single ``SubQuery`` wrapping the original query if the
    LLM response cannot be parsed or yields no valid sub-queries.
    """
    research_config = config["configurable"]["research_config"]
    original_query: str = state["original_query"]

    llm = make_llm(
        model=research_config.default_model,
        hf_token=research_config.hf_token,
        max_new_tokens=research_config.llm_max_tokens,
        provider=research_config.llm_provider,
        temperature=research_config.llm_temperature,
        top_p=research_config.llm_top_p,
        top_k=research_config.llm_top_k,
    )

    response = await llm.ainvoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=original_query),
        ]
    )

    sub_queries = _parse_sub_queries(response.content, original_query)
    _log.debug(
        "query_planner produced %d sub-queries for %r",
        len(sub_queries),
        original_query[:60],
    )
    return {"sub_queries": sub_queries}


def _parse_sub_queries(content: str, fallback_query: str) -> list[SubQuery]:
    """Parse LLM output into ``SubQuery`` objects with a safe fallback."""
    try:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            raise ValueError("no JSON array found in response")
        items = json.loads(match.group())
        sub_queries = [
            SubQuery(query=item["query"], domain=item.get("domain", ""))
            for item in items
            if item.get("query", "").strip()
        ]
        if sub_queries:
            return sub_queries
    except Exception as exc:  # noqa: BLE001
        _log.warning("query_planner parse failed (%s); using fallback", exc)
    return [SubQuery(query=fallback_query)]
