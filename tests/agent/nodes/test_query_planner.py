"""Tests for mara/agent/nodes/query_planner.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agent.nodes.query_planner import _build_system_prompt, _parse_sub_queries, query_planner_node
from mara.agents.types import SubQuery


# ---------------------------------------------------------------------------
# _parse_sub_queries — unit tests (no LLM needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, expected_queries",
    [
        (
            '[{"query": "sub one", "domain": "empirical"}, {"query": "sub two", "domain": ""}]',
            ["sub one", "sub two"],
        ),
        (
            'Some preamble\n[{"query": "extracted", "domain": ""}]\ntrailing',
            ["extracted"],
        ),
        (
            '[{"query": "  trimmed  ", "domain": "review"}]',
            ["trimmed"],
        ),
    ],
    ids=["clean_json", "embedded_in_text", "whitespace_query"],
)
def test_parse_sub_queries_valid(content: str, expected_queries: list[str]) -> None:
    result = _parse_sub_queries(content, "fallback")
    assert [sq.query for sq in result] == expected_queries


def test_parse_sub_queries_parses_agent_field() -> None:
    content = '[{"query": "quantum error correction", "domain": "theoretical", "agent": "arxiv"}]'
    result = _parse_sub_queries(content, "fallback")
    assert len(result) == 1
    assert result[0].agent == "arxiv"


def test_parse_sub_queries_agent_defaults_to_empty() -> None:
    content = '[{"query": "some query", "domain": ""}]'
    result = _parse_sub_queries(content, "fallback")
    assert result[0].agent == ""


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "[]",
        '[{"query": "", "domain": ""}]',
        '[{"no_query_key": "x"}]',
    ],
    ids=["no_json", "empty_array", "empty_query", "missing_query_key"],
)
def test_parse_sub_queries_fallback(content: str) -> None:
    result = _parse_sub_queries(content, "fallback query")
    assert len(result) == 1
    assert result[0].query == "fallback query"


def test_parse_sub_queries_invalid_json() -> None:
    result = _parse_sub_queries("[{broken json", "fallback")
    assert result == [SubQuery(query="fallback")]


# ---------------------------------------------------------------------------
# query_planner_node — async node tests with mocked LLM
# ---------------------------------------------------------------------------


def _make_llm_mock(response_content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = response_content
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


def test_build_system_prompt_includes_agent_roster() -> None:
    """The system prompt must contain the live registry summary."""
    with patch(
        "mara.agent.nodes.query_planner.get_registry_summary",
        return_value="[fake] Does something.",
    ):
        prompt = _build_system_prompt()
    assert "[fake] Does something." in prompt
    assert "agent" in prompt.lower()


async def test_query_planner_node_valid_response(runnable_config) -> None:
    payload = json.dumps([
        {"query": "climate change effects", "domain": "empirical", "agent": "pubmed"},
        {"query": "mitigation strategies", "domain": "policy", "agent": "web"},
    ])
    mock_llm = _make_llm_mock(payload)

    with patch("mara.agent.nodes.query_planner.make_llm", return_value=mock_llm):
        result = await query_planner_node(
            {"original_query": "climate change"}, runnable_config
        )

    sub_queries = result["sub_queries"]
    assert len(sub_queries) == 2
    assert sub_queries[0].query == "climate change effects"
    assert sub_queries[0].domain == "empirical"
    assert sub_queries[0].agent == "pubmed"
    assert sub_queries[1].query == "mitigation strategies"
    assert sub_queries[1].agent == "web"


async def test_query_planner_node_fallback_on_bad_json(runnable_config) -> None:
    mock_llm = _make_llm_mock("I cannot decompose this.")

    with patch("mara.agent.nodes.query_planner.make_llm", return_value=mock_llm):
        result = await query_planner_node(
            {"original_query": "original question"}, runnable_config
        )

    sub_queries = result["sub_queries"]
    assert len(sub_queries) == 1
    assert sub_queries[0].query == "original question"


async def test_query_planner_node_calls_make_llm_with_config(runnable_config) -> None:
    mock_llm = _make_llm_mock('[{"query": "sub", "domain": ""}]')

    with patch("mara.agent.nodes.query_planner.make_llm", return_value=mock_llm) as mk:
        await query_planner_node({"original_query": "q"}, runnable_config)

    rc = runnable_config["configurable"]["research_config"]
    mk.assert_called_once_with(
        model=rc.default_model,
        hf_token=rc.hf_token,
        max_new_tokens=rc.llm_max_tokens,
        provider=rc.llm_provider,
        temperature=rc.llm_temperature,
        top_p=rc.llm_top_p,
        top_k=rc.llm_top_k,
    )
