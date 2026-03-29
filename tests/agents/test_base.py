"""Tests for mara/agents/base.py — SpecialistAgent."""

from unittest.mock import AsyncMock

import pytest

from mara.agents.base import SpecialistAgent
from mara.agents.registry import agent
from mara.agents.types import AgentFindings, RawChunk, SubQuery
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-03-28T00:00:00+00:00"

_REQUIRED = {
    "brave_api_key": "b",
    "hf_token": "h",
    "firecrawl_api_key": "f",
    "core_api_key": "c",
    "s2_api_key": "s",
    "ncbi_api_key": "n",
}


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
        async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
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
            async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
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
            async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
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


# ---------------------------------------------------------------------------
# _chunk()
# ---------------------------------------------------------------------------


class TestChunk:
    def test_empty_list_returns_empty(self, config):
        instance = _make_concrete("chunk_empty_agent")(config)
        assert instance._chunk([]) == []

    def test_short_text_returned_unchanged(self, config):
        instance = _make_concrete("chunk_short_agent")(config)
        chunk = _raw(text="hello")
        assert instance._chunk([chunk]) == [chunk]

    def test_long_text_is_split(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("chunk_split_agent")(cfg)
        result = instance._chunk([_raw(text="0123456789")])
        assert len(result) == 2
        assert result[0].text == "01234"
        assert result[1].text == "56789"

    def test_overlap_creates_overlapping_windows(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=2)
        instance = _make_concrete("chunk_overlap_agent")(cfg)
        result = instance._chunk([_raw(text="0123456789")])
        # step = 5 - 2 = 3: windows at 0, 3, 6
        assert result[0].text == "01234"
        assert result[1].text == "34567"
        assert result[2].text == "6789"

    def test_split_preserves_url_and_metadata(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("chunk_meta_agent")(cfg)
        chunk = _raw(url="https://my.com", text="0123456789", source_type="latex")
        result = instance._chunk([chunk])
        assert all(c.url == "https://my.com" for c in result)
        assert all(c.source_type == "latex" for c in result)

    def test_multiple_chunks_handled_independently(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("chunk_multi_agent")(cfg)
        c1 = _raw(url="https://a.com", text="ABCDE")  # exactly size — no split
        c2 = _raw(url="https://b.com", text="0123456789")  # split into 2
        result = instance._chunk([c1, c2])
        assert len(result) == 3

    def test_size_zero_returns_raw(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=0, chunk_overlap=0)
        instance = _make_concrete("chunk_zero_agent")(cfg)
        chunks = [_raw()]
        assert instance._chunk(chunks) == chunks

    def test_overlap_gte_size_returns_raw(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=5)
        instance = _make_concrete("chunk_degen_agent")(cfg)
        chunks = [_raw(text="hello world")]
        assert instance._chunk(chunks) == chunks


# ---------------------------------------------------------------------------
# _filter()
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_applies_per_url_cap(self, config):
        # Default CapFilter: max_chunks_per_url=3
        instance = _make_concrete("filter_cap_agent")(config)
        chunks = [_raw(url="https://same.com", text=f"text {i}") for i in range(5)]
        result = instance._filter(chunks, "q")
        assert len(result) == 3

    def test_filter_passes_chunks_within_limits(self, config):
        instance = _make_concrete("filter_pass_agent")(config)
        chunks = [_raw(url=f"https://example.com/{i}") for i in range(3)]
        result = instance._filter(chunks, "q")
        assert len(result) == 3

    def test_filter_passes_query_to_chunk_filter(self):
        from unittest.mock import MagicMock
        mock_filter = MagicMock()
        mock_filter.filter.return_value = []
        cfg = ResearchConfig(**_REQUIRED, chunk_filter=mock_filter)
        instance = _make_concrete("filter_mock_agent")(cfg)
        chunks = [_raw()]
        instance._filter(chunks, "my query")
        mock_filter.filter.assert_called_once_with(chunks, "my query")


# ---------------------------------------------------------------------------
# _retrieve() pipeline
# ---------------------------------------------------------------------------


class TestRetrievePipeline:
    async def test_retrieve_calls_search_and_returns_list(self, config):
        instance = _make_concrete("pipeline_agent")(config)
        instance._search = AsyncMock(return_value=[_raw()])
        result = await instance._retrieve(SubQuery(query="q"))
        instance._search.assert_awaited_once()
        assert isinstance(result, list)

    async def test_retrieve_passes_sub_query_to_search(self, config):
        instance = _make_concrete("pipeline_sq_agent")(config)
        instance._search = AsyncMock(return_value=[])
        sq = SubQuery(query="specific")
        await instance._retrieve(sq)
        instance._search.assert_awaited_once_with(sq)

    async def test_retrieve_applies_chunking(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("pipeline_chunk_agent")(cfg)
        # 10-char text splits into 2 chunks of size 5
        instance._search = AsyncMock(return_value=[_raw(text="0123456789")])
        result = await instance._retrieve(SubQuery(query="q"))
        assert len(result) == 2

    async def test_retrieve_applies_filtering(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=1000, chunk_overlap=0)
        instance = _make_concrete("pipeline_filter_agent")(cfg)
        # Default CapFilter: max_chunks_per_url=3 — return 5 from same URL, expect 3
        instance._search = AsyncMock(
            return_value=[_raw(url="https://same.com", text=f"text {i}") for i in range(5)]
        )
        result = await instance._retrieve(SubQuery(query="q"))
        assert len(result) == 3

    async def test_retrieve_empty_search_returns_empty(self, config):
        instance = _make_concrete("pipeline_empty_agent")(config)
        instance._search = AsyncMock(return_value=[])
        result = await instance._retrieve(SubQuery(query="q"))
        assert result == []
