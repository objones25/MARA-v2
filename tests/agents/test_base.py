"""Tests for mara/agents/base.py — SpecialistAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from mara.agents.base import SpecialistAgent
from mara.agents.registry import agent
from mara.agents.types import AgentFindings, RawChunk, SubQuery
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-03-28T00:00:00+00:00"


def _raw(
    url: str = "https://example.com",
    text: str = "hello",
    retrieved_at: str = _TS,
    source_type: str = "abstract",
    sub_query: str = "foo",
) -> RawChunk:
    return RawChunk(
        url=url,
        text=text,
        retrieved_at=retrieved_at,
        source_type=source_type,
        sub_query=sub_query,
    )


def _make_concrete(name: str) -> type[SpecialistAgent]:
    """Register a no-op concrete SpecialistAgent subclass."""

    @agent(name)
    class ConcreteAgent(SpecialistAgent):
        async def _retrieve(self, sub_query: SubQuery) -> list[RawChunk]:
            return []  # overridden in tests via mock

    ConcreteAgent.__name__ = name
    return ConcreteAgent


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_stores_config(self, config):
        cls = _make_concrete("init_agent")
        agent_instance = cls(config)
        assert agent_instance.config is config


# ---------------------------------------------------------------------------
# _agent_type()
# ---------------------------------------------------------------------------


class TestAgentType:
    def test_returns_registered_name(self, config):
        cls = _make_concrete("my_agent")
        instance = cls(config)
        assert instance._agent_type() == "my_agent"

    def test_different_agents_return_different_types(self, config):
        cls_a = _make_concrete("agent_a")
        cls_b = _make_concrete("agent_b")
        assert cls_a(config)._agent_type() == "agent_a"
        assert cls_b(config)._agent_type() == "agent_b"

    def test_unregistered_agent_raises(self, config):
        class Unregistered(SpecialistAgent):
            async def _retrieve(self, sub_query: SubQuery) -> list[RawChunk]:
                return []

        instance = Unregistered(config)
        with pytest.raises(StopIteration):
            instance._agent_type()


# ---------------------------------------------------------------------------
# model()
# ---------------------------------------------------------------------------


class TestModel:
    def test_returns_default_model_when_no_override(self, config):
        cls = _make_concrete("model_agent")
        instance = cls(config)
        assert instance.model() == config.default_model

    def test_returns_override_when_set(self, config):
        from mara.config import ResearchConfig

        overridden_config = ResearchConfig(
            **{
                "brave_api_key": "b",
                "hf_token": "h",
                "firecrawl_api_key": "f",
                "core_api_key": "c",
                "s2_api_key": "s",
                "ncbi_api_key": "n",
                "model_overrides": {"override_agent": "smaller-model"},
            }
        )

        @agent("override_agent")
        class OverrideAgent(SpecialistAgent):
            async def _retrieve(self, sub_query: SubQuery) -> list[RawChunk]:
                return []

        instance = OverrideAgent(overridden_config)
        assert instance.model() == "smaller-model"

    def test_override_for_other_agent_does_not_affect_this_one(self, config):
        from mara.config import ResearchConfig

        cfg = ResearchConfig(
            **{
                "brave_api_key": "b",
                "hf_token": "h",
                "firecrawl_api_key": "f",
                "core_api_key": "c",
                "s2_api_key": "s",
                "ncbi_api_key": "n",
                "model_overrides": {"other_agent": "tiny-model"},
            }
        )
        cls = _make_concrete("unaffected_agent")
        instance = cls(cfg)
        assert instance.model() == cfg.default_model


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    async def test_run_returns_agent_findings(self, config):
        cls = _make_concrete("run_agent")
        instance = cls(config)
        raw = [_raw()]
        instance._retrieve = AsyncMock(return_value=raw)

        result = await instance.run(SubQuery(query="test query"))

        assert isinstance(result, AgentFindings)

    async def test_run_sets_correct_agent_type(self, config):
        cls = _make_concrete("type_check_agent")
        instance = cls(config)
        instance._retrieve = AsyncMock(return_value=[_raw()])

        result = await instance.run(SubQuery(query="q"))
        assert result.agent_type == "type_check_agent"

    async def test_run_sets_query(self, config):
        cls = _make_concrete("query_agent")
        instance = cls(config)
        instance._retrieve = AsyncMock(return_value=[_raw()])

        result = await instance.run(SubQuery(query="my query"))
        assert result.query == "my query"

    async def test_run_hashes_chunks_correctly(self, config):
        cls = _make_concrete("hash_agent")
        instance = cls(config)
        raw = _raw(url="https://arxiv.org/abs/1", text="content", retrieved_at=_TS)
        instance._retrieve = AsyncMock(return_value=[raw])

        result = await instance.run(SubQuery(query="q"))

        expected_hash = hash_chunk(raw.url, raw.text, raw.retrieved_at, "sha256")
        assert result.chunks[0].hash == expected_hash

    async def test_run_assigns_chunk_index(self, config):
        cls = _make_concrete("index_agent")
        instance = cls(config)
        raws = [_raw(url=f"https://example.com/{i}") for i in range(3)]
        instance._retrieve = AsyncMock(return_value=raws)

        result = await instance.run(SubQuery(query="q"))
        assert [c.chunk_index for c in result.chunks] == [0, 1, 2]

    async def test_run_empty_retrieve_returns_empty_findings(self, config):
        cls = _make_concrete("empty_agent")
        instance = cls(config)
        instance._retrieve = AsyncMock(return_value=[])

        result = await instance.run(SubQuery(query="q"))
        assert result.chunks == ()
        assert result.merkle_root == ""

    async def test_run_merkle_root_verified(self, config):
        cls = _make_concrete("merkle_agent")
        instance = cls(config)
        raws = [_raw(url=f"https://example.com/{i}") for i in range(4)]
        instance._retrieve = AsyncMock(return_value=raws)

        result = await instance.run(SubQuery(query="q"))

        expected_hashes = [
            hash_chunk(r.url, r.text, r.retrieved_at, "sha256") for r in raws
        ]
        expected_root = build_merkle_tree(expected_hashes, "sha256").root
        assert result.merkle_root == expected_root

    async def test_run_uses_config_hash_algorithm(self, config):
        cls = _make_concrete("algo_agent")
        instance = cls(config)
        raw = _raw()
        instance._retrieve = AsyncMock(return_value=[raw])

        result = await instance.run(SubQuery(query="q"))
        # sha256 hex digest is 64 chars
        assert len(result.chunks[0].hash) == 64

    async def test_run_passes_sub_query_to_retrieve(self, config):
        cls = _make_concrete("pass_query_agent")
        instance = cls(config)
        instance._retrieve = AsyncMock(return_value=[])

        sq = SubQuery(query="specific query")
        await instance.run(sq)
        instance._retrieve.assert_awaited_once_with(sq)
