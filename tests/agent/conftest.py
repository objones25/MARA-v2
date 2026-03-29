"""Shared fixtures and helpers for agent pipeline tests."""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.runnables import RunnableConfig

from mara.agents.types import AgentFindings, SubQuery, VerifiedChunk
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree

_REQUIRED = {
    "brave_api_key": "brave-key",
    "hf_token": "hf-token",
    "firecrawl_api_key": "fc-key",
    "core_api_key": "core-key",
    "s2_api_key": "s2-key",
    "ncbi_api_key": "ncbi-key",
}


@pytest.fixture()
def research_config() -> ResearchConfig:
    return ResearchConfig(**_REQUIRED, _env_file=None)


@pytest.fixture()
def runnable_config(research_config: ResearchConfig) -> RunnableConfig:
    return {"configurable": {"research_config": research_config}}


def make_chunk(
    url: str = "https://example.com/doc",
    text: str = "sample source text",
    retrieved_at: str = "2024-01-01T00:00:00Z",
    source_type: str = "test",
    sub_query: str = "test query",
    chunk_index: int = 0,
) -> VerifiedChunk:
    h = hash_chunk(url, text, retrieved_at, "sha256")
    return VerifiedChunk(
        hash=h,
        url=url,
        text=text,
        retrieved_at=retrieved_at,
        source_type=source_type,
        sub_query=sub_query,
        chunk_index=chunk_index,
    )


def make_findings(
    agent_type: str = "test_agent",
    query: str = "test query",
    chunks: tuple[VerifiedChunk, ...] | None = None,
    algorithm: str = "sha256",
) -> AgentFindings:
    if chunks is None:
        chunks = (make_chunk(),)
    tree = build_merkle_tree([c.hash for c in chunks], algorithm)
    return AgentFindings(
        agent_type=agent_type,
        query=query,
        chunks=chunks,
        merkle_root=tree.root,
        merkle_tree=tree,
    )


@pytest.fixture()
def sample_findings() -> AgentFindings:
    return make_findings()
