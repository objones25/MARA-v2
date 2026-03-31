"""Tests for mara/agent/nodes/report_synthesizer.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mara.agent.nodes.report_synthesizer import report_synthesizer_node
from tests.agent.conftest import make_chunk


def _make_llm_mock(response_text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = response_text
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


async def test_report_synthesizer_returns_report_string(runnable_config) -> None:
    chunk = make_chunk(text="source content", chunk_index=0)
    state = {
        "original_query": "test query",
        "flattened_chunks": [chunk],
    }
    mock_llm = _make_llm_mock("This is the report.")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm):
        result = await report_synthesizer_node(state, runnable_config)

    assert result == {"report": "This is the report."}


async def test_report_synthesizer_empty_chunks(runnable_config) -> None:
    state = {"original_query": "question", "flattened_chunks": []}
    mock_llm = _make_llm_mock("Empty report.")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm):
        result = await report_synthesizer_node(state, runnable_config)

    assert result["report"] == "Empty report."


async def test_report_synthesizer_missing_chunks_key(runnable_config) -> None:
    """Missing flattened_chunks should default to [] without raising."""
    state = {"original_query": "question"}
    mock_llm = _make_llm_mock("Still works.")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm):
        result = await report_synthesizer_node(state, runnable_config)

    assert result["report"] == "Still works."


async def test_report_synthesizer_includes_chunk_citations_in_prompt(runnable_config) -> None:
    chunk = make_chunk(text="important finding", chunk_index=3)
    state = {"original_query": "q", "flattened_chunks": [chunk]}
    mock_llm = _make_llm_mock("report")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm):
        await report_synthesizer_node(state, runnable_config)

    call_args = mock_llm.ainvoke.call_args[0][0]
    human_msg_content = call_args[-1].content
    # Citation format: [index:short_hash]
    assert f"[3:{chunk.short_hash}]" in human_msg_content
    assert "important finding" in human_msg_content


async def test_report_synthesizer_includes_sub_query_distribution(runnable_config) -> None:
    """Sub-query coverage block appears in the prompt when chunks carry sub_queries."""
    chunks = [
        make_chunk(text="content A", sub_query="alpha topic", chunk_index=0),
        make_chunk(
            text="content B",
            sub_query="beta topic",
            chunk_index=1,
            url="https://example.com/doc2",
        ),
        make_chunk(
            text="content C",
            sub_query="alpha topic",
            chunk_index=2,
            url="https://example.com/doc3",
        ),
    ]
    state = {"original_query": "q", "flattened_chunks": chunks}
    mock_llm = _make_llm_mock("report")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm):
        await report_synthesizer_node(state, runnable_config)

    call_args = mock_llm.ainvoke.call_args[0][0]
    human_msg_content = call_args[-1].content
    assert "Sub-query coverage" in human_msg_content
    assert "alpha topic: 2 chunks" in human_msg_content
    assert "beta topic: 1 chunk" in human_msg_content


async def test_report_synthesizer_no_sub_query_distribution_when_empty(runnable_config) -> None:
    """No distribution block when all chunks have empty sub_query strings."""
    chunk = make_chunk(text="content", sub_query="", chunk_index=0)
    state = {"original_query": "q", "flattened_chunks": [chunk]}
    mock_llm = _make_llm_mock("report")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm):
        await report_synthesizer_node(state, runnable_config)

    call_args = mock_llm.ainvoke.call_args[0][0]
    human_msg_content = call_args[-1].content
    assert "Sub-query coverage" not in human_msg_content


async def test_report_synthesizer_calls_make_llm_with_config(runnable_config) -> None:
    state = {"original_query": "q", "flattened_chunks": []}
    mock_llm = _make_llm_mock("ok")

    with patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm) as mk:
        await report_synthesizer_node(state, runnable_config)

    rc = runnable_config["configurable"]["research_config"]
    mk.assert_called_once_with(
        model=rc.default_model,
        hf_token=rc.hf_token,
        max_new_tokens=rc.llm_max_tokens,
        provider=rc.llm_provider,
        temperature=rc.synthesizer_temperature,
        top_p=rc.llm_top_p,
        top_k=rc.llm_top_k,
    )
