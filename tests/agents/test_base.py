"""Tests for mara/agents/base.py — SpecialistAgent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mara.agents.base import SpecialistAgent
from mara.agents.registry import AgentConfig, agent
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
        agent_instance = cls(config, AgentConfig())
        assert agent_instance.config is config

    def test_repr_contains_class_name(self, config):
        cls = _make_concrete("repr_agent")
        instance = cls(config, AgentConfig())
        assert "ConcreteAgent" in repr(instance) or "repr_agent" in repr(instance)

    def test_str_returns_agent_type(self, config):
        cls = _make_concrete("str_agent")
        instance = cls(config, AgentConfig())
        assert str(instance) == "str_agent"


# ---------------------------------------------------------------------------
# _agent_type()
# ---------------------------------------------------------------------------


class TestAgentType:
    def test_returns_registered_name(self, config):
        cls = _make_concrete("my_agent")
        instance = cls(config, AgentConfig())
        assert instance._agent_type() == "my_agent"

    def test_different_agents_return_different_types(self, config):
        cls_a = _make_concrete("agent_a")
        cls_b = _make_concrete("agent_b")
        assert cls_a(config, AgentConfig())._agent_type() == "agent_a"
        assert cls_b(config, AgentConfig())._agent_type() == "agent_b"

    def test_unregistered_agent_raises(self, config):
        class Unregistered(SpecialistAgent):
            async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
                return []

        with pytest.raises(StopIteration):
            Unregistered(config, AgentConfig())


# ---------------------------------------------------------------------------
# model()
# ---------------------------------------------------------------------------


class TestModel:
    def test_returns_default_model_when_no_override(self, config):
        cls = _make_concrete("model_agent")
        instance = cls(config, AgentConfig())
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

        instance = OverrideAgent(overridden_config, AgentConfig())
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
        instance = cls(cfg, AgentConfig())
        assert instance.model() == cfg.default_model


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    async def test_run_returns_agent_findings(self, config):
        cls = _make_concrete("run_agent")
        instance = cls(config, AgentConfig())
        raw = [_raw()]
        instance._retrieve = AsyncMock(return_value=raw)

        result = await instance.run(SubQuery(query="test query"))

        assert isinstance(result, AgentFindings)

    async def test_run_sets_correct_agent_type(self, config):
        cls = _make_concrete("type_check_agent")
        instance = cls(config, AgentConfig())
        instance._retrieve = AsyncMock(return_value=[_raw()])

        result = await instance.run(SubQuery(query="q"))
        assert result.agent_type == "type_check_agent"

    async def test_run_sets_query(self, config):
        cls = _make_concrete("query_agent")
        instance = cls(config, AgentConfig())
        instance._retrieve = AsyncMock(return_value=[_raw()])

        result = await instance.run(SubQuery(query="my query"))
        assert result.query == "my query"

    async def test_run_hashes_chunks_correctly(self, config):
        cls = _make_concrete("hash_agent")
        instance = cls(config, AgentConfig())
        raw = _raw(url="https://arxiv.org/abs/1", text="content", retrieved_at=_TS)
        instance._retrieve = AsyncMock(return_value=[raw])

        result = await instance.run(SubQuery(query="q"))

        expected_hash = hash_chunk(raw.url, raw.text, raw.retrieved_at, "sha256")
        assert result.chunks[0].hash == expected_hash

    async def test_run_assigns_chunk_index(self, config):
        cls = _make_concrete("index_agent")
        instance = cls(config, AgentConfig())
        raws = [_raw(url=f"https://example.com/{i}") for i in range(3)]
        instance._retrieve = AsyncMock(return_value=raws)

        result = await instance.run(SubQuery(query="q"))
        assert [c.chunk_index for c in result.chunks] == [0, 1, 2]

    async def test_run_empty_retrieve_returns_empty_findings(self, config):
        cls = _make_concrete("empty_agent")
        instance = cls(config, AgentConfig())
        instance._retrieve = AsyncMock(return_value=[])

        result = await instance.run(SubQuery(query="q"))
        assert result.chunks == ()
        assert result.merkle_root == ""

    async def test_run_merkle_root_verified(self, config):
        cls = _make_concrete("merkle_agent")
        instance = cls(config, AgentConfig())
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
        instance = cls(config, AgentConfig())
        raw = _raw()
        instance._retrieve = AsyncMock(return_value=[raw])

        result = await instance.run(SubQuery(query="q"))
        # sha256 hex digest is 64 chars
        assert len(result.chunks[0].hash) == 64

    async def test_run_passes_sub_query_to_retrieve(self, config):
        cls = _make_concrete("pass_query_agent")
        instance = cls(config, AgentConfig())
        instance._retrieve = AsyncMock(return_value=[])

        sq = SubQuery(query="specific query")
        await instance.run(sq)
        instance._retrieve.assert_awaited_once_with(sq)

    async def test_run_wraps_retrieve_exception_with_context(self, config):
        cls = _make_concrete("run_exc_agent")
        instance = cls(config, AgentConfig())
        instance._retrieve = AsyncMock(side_effect=ValueError("network down"))

        with pytest.raises(RuntimeError, match="run_exc_agent") as exc_info:
            await instance.run(SubQuery(query="q"))
        assert isinstance(exc_info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# _chunk()
# ---------------------------------------------------------------------------


class TestChunk:
    def test_empty_list_returns_empty(self, config):
        instance = _make_concrete("chunk_empty_agent")(config, AgentConfig())
        assert instance._chunk([]) == []

    def test_short_text_returned_unchanged(self, config):
        instance = _make_concrete("chunk_short_agent")(config, AgentConfig())
        chunk = _raw(text="hello")
        assert instance._chunk([chunk]) == [chunk]

    def test_long_text_is_split(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("chunk_split_agent")(cfg, AgentConfig())
        result = instance._chunk([_raw(text="0123456789")])
        assert len(result) == 2
        assert result[0].text == "01234"
        assert result[1].text == "56789"

    def test_overlap_creates_overlapping_windows(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=2)
        instance = _make_concrete("chunk_overlap_agent")(cfg, AgentConfig())
        result = instance._chunk([_raw(text="0123456789")])
        # step = 5 - 2 = 3: windows at 0, 3, 6
        assert result[0].text == "01234"
        assert result[1].text == "34567"
        assert result[2].text == "6789"

    def test_split_preserves_url_and_metadata(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("chunk_meta_agent")(cfg, AgentConfig())
        chunk = _raw(url="https://my.com", text="0123456789", source_type="latex")
        result = instance._chunk([chunk])
        assert all(c.url == "https://my.com" for c in result)
        assert all(c.source_type == "latex" for c in result)

    def test_multiple_chunks_handled_independently(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("chunk_multi_agent")(cfg, AgentConfig())
        c1 = _raw(url="https://a.com", text="ABCDE")  # exactly size — no split
        c2 = _raw(url="https://b.com", text="0123456789")  # split into 2
        result = instance._chunk([c1, c2])
        assert len(result) == 3

    def test_size_zero_returns_raw(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=0, chunk_overlap=0)
        instance = _make_concrete("chunk_zero_agent")(cfg, AgentConfig())
        chunks = [_raw()]
        assert instance._chunk(chunks) == chunks

    def test_overlap_gte_size_returns_raw(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=5)
        instance = _make_concrete("chunk_degen_agent")(cfg, AgentConfig())
        chunks = [_raw(text="hello world")]
        assert instance._chunk(chunks) == chunks


# ---------------------------------------------------------------------------
# _filter()
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_applies_per_url_cap(self, config):
        # Default CapFilter: max_chunks_per_url=3
        instance = _make_concrete("filter_cap_agent")(config, AgentConfig())
        chunks = [_raw(url="https://same.com", text=f"text {i}") for i in range(5)]
        result = instance._filter(chunks, "q")
        assert len(result) == 3

    def test_filter_passes_chunks_within_limits(self, config):
        instance = _make_concrete("filter_pass_agent")(config, AgentConfig())
        chunks = [_raw(url=f"https://example.com/{i}") for i in range(3)]
        result = instance._filter(chunks, "q")
        assert len(result) == 3

    def test_filter_passes_query_to_chunk_filter(self):
        from unittest.mock import MagicMock
        mock_filter = MagicMock()
        mock_filter.filter.return_value = []
        cfg = ResearchConfig(**_REQUIRED, chunk_filter=mock_filter)
        instance = _make_concrete("filter_mock_agent")(cfg, AgentConfig())
        chunks = [_raw()]
        instance._filter(chunks, "my query")
        mock_filter.filter.assert_called_once_with(chunks, "my query")


# ---------------------------------------------------------------------------
# _retrieve() pipeline
# ---------------------------------------------------------------------------


class TestRetrievePipeline:
    async def test_retrieve_calls_search_and_returns_list(self, config):
        instance = _make_concrete("pipeline_agent")(config, AgentConfig())
        instance._search = AsyncMock(return_value=[_raw()])
        result = await instance._retrieve(SubQuery(query="q"))
        instance._search.assert_awaited_once()
        assert isinstance(result, list)

    async def test_retrieve_passes_sub_query_to_search(self, config):
        instance = _make_concrete("pipeline_sq_agent")(config, AgentConfig())
        instance._search = AsyncMock(return_value=[])
        sq = SubQuery(query="specific")
        await instance._retrieve(sq)
        instance._search.assert_awaited_once_with(sq)

    async def test_retrieve_applies_chunking(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=5, chunk_overlap=0)
        instance = _make_concrete("pipeline_chunk_agent")(cfg, AgentConfig())
        # 10-char text splits into 2 chunks of size 5
        instance._search = AsyncMock(return_value=[_raw(text="0123456789")])
        result = await instance._retrieve(SubQuery(query="q"))
        assert len(result) == 2

    async def test_retrieve_applies_filtering(self):
        cfg = ResearchConfig(**_REQUIRED, chunk_size=1000, chunk_overlap=0)
        instance = _make_concrete("pipeline_filter_agent")(cfg, AgentConfig())
        # Default CapFilter: max_chunks_per_url=3 — return 5 from same URL, expect 3
        instance._search = AsyncMock(
            return_value=[_raw(url="https://same.com", text=f"text {i}") for i in range(5)]
        )
        result = await instance._retrieve(SubQuery(query="q"))
        assert len(result) == 3

    async def test_retrieve_empty_search_returns_empty(self, config):
        instance = _make_concrete("pipeline_empty_agent")(config, AgentConfig())
        instance._search = AsyncMock(return_value=[])
        result = await instance._retrieve(SubQuery(query="q"))
        assert result == []


# ---------------------------------------------------------------------------
# _fetch_with_retry()
# ---------------------------------------------------------------------------


def _http_error(status_code: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.HTTPStatusError("error", request=MagicMock(), response=resp)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    SpecialistAgent._reset_rate_limit_state()
    yield
    SpecialistAgent._reset_rate_limit_state()


class TestFetchWithRetry:
    async def test_success_returns_chunks(self, config):
        instance = _make_concrete("fwr_ok")(config, AgentConfig())
        chunks = [_raw()]
        instance._search = AsyncMock(return_value=chunks)
        result = await instance._fetch_with_retry(SubQuery(query="q"))
        assert result == chunks

    async def test_404_returns_empty(self, config):
        instance = _make_concrete("fwr_404")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(404))
        result = await instance._fetch_with_retry(SubQuery(query="q"))
        assert result == []

    async def test_401_raises_immediately(self, config):
        instance = _make_concrete("fwr_401")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(401))
        with pytest.raises(httpx.HTTPStatusError):
            await instance._fetch_with_retry(SubQuery(query="q"))
        assert instance._search.await_count == 1

    async def test_403_raises_immediately(self, config):
        instance = _make_concrete("fwr_403")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(403))
        with pytest.raises(httpx.HTTPStatusError):
            await instance._fetch_with_retry(SubQuery(query="q"))
        assert instance._search.await_count == 1

    async def test_429_retries_until_max_retries_then_raises(self, config):
        instance = _make_concrete("fwr_429")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(429))
        with patch("mara.agents.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        assert instance._search.await_count == config.max_retries + 1

    async def test_5xx_retries_until_max_retries_then_raises(self, config):
        instance = _make_concrete("fwr_500")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(500))
        with patch("mara.agents.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        assert instance._search.await_count == config.max_retries + 1

    async def test_connect_error_retries_then_raises(self, config):
        instance = _make_concrete("fwr_conn")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("mara.agents.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.ConnectError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        assert instance._search.await_count == config.max_retries + 1

    async def test_timeout_retries_then_raises(self, config):
        instance = _make_concrete("fwr_timeout")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        with patch("mara.agents.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.TimeoutException):
                await instance._fetch_with_retry(SubQuery(query="q"))
        assert instance._search.await_count == config.max_retries + 1

    async def test_backoff_sleep_values_match_config(self, config):
        instance = _make_concrete("fwr_backoff")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(503))
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        expected = [
            ((config.retry_backoff_base**i,), {})
            for i in range(config.max_retries)
        ]
        actual = [(call.args, call.kwargs) for call in mock_sleep.call_args_list]
        assert actual == expected

    async def test_succeeds_after_transient_failure(self, config):
        instance = _make_concrete("fwr_transient")(config, AgentConfig())
        ok_chunk = _raw()
        instance._search = AsyncMock(
            side_effect=[_http_error(429), [ok_chunk]]
        )
        with patch("mara.agents.base.asyncio.sleep", new_callable=AsyncMock):
            result = await instance._fetch_with_retry(SubQuery(query="q"))
        assert result == [ok_chunk]
        assert instance._search.await_count == 2

    async def test_cache_hit_skips_search(self):
        from mara.agents.cache import InMemoryCache
        from mara.config import ResearchConfig

        cache = InMemoryCache()
        cfg = ResearchConfig(**_REQUIRED, search_cache=cache, _env_file=None)
        cls = _make_concrete("fwr_cache_hit")
        instance = cls(cfg, AgentConfig())
        cached_chunks = [_raw(text="cached")]
        await cache.set("fwr_cache_hit", "q", cached_chunks)

        instance._search = AsyncMock(return_value=[_raw(text="fresh")])
        result = await instance._fetch_with_retry(SubQuery(query="q"))

        assert result == cached_chunks
        instance._search.assert_not_awaited()

    async def test_cache_miss_populates_cache(self):
        from mara.agents.cache import InMemoryCache
        from mara.config import ResearchConfig

        cache = InMemoryCache()
        cfg = ResearchConfig(**_REQUIRED, search_cache=cache, _env_file=None)
        cls = _make_concrete("fwr_cache_miss")
        instance = cls(cfg, AgentConfig())
        fresh = [_raw(text="from network")]
        instance._search = AsyncMock(return_value=fresh)

        await instance._fetch_with_retry(SubQuery(query="q"))

        stored = await cache.get("fwr_cache_miss", "q")
        assert stored == fresh

    async def test_cache_not_populated_on_error(self):
        from mara.agents.cache import InMemoryCache
        from mara.config import ResearchConfig

        cache = InMemoryCache()
        cfg = ResearchConfig(**_REQUIRED, search_cache=cache, _env_file=None)
        cls = _make_concrete("fwr_cache_err")
        instance = cls(cfg, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(500))

        with patch("mara.agents.base.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))

        assert await cache.get("fwr_cache_err", "q") is None

    async def test_retry_after_header_used_when_present(self, config):
        instance = _make_concrete("fwr_ra_present")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(429, retry_after="60"))
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        # First sleep must use the Retry-After value, not the normal backoff
        assert mock_sleep.call_args_list[0] == ((60.0,), {})

    async def test_retry_after_header_invalid_falls_back_to_backoff(self, config):
        instance = _make_concrete("fwr_ra_invalid")(config, AgentConfig())
        instance._search = AsyncMock(side_effect=_http_error(429, retry_after="not-a-number"))
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        expected = [config.retry_backoff_base**i for i in range(config.max_retries)]
        actual = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual == pytest.approx(expected)

    async def test_agent_config_retry_backoff_base_overrides_config(self, config):
        agent_cfg = AgentConfig(retry_backoff_base=3.0)
        instance = _make_concrete("fwr_cfg_backoff")(config, agent_cfg)
        instance._search = AsyncMock(side_effect=_http_error(503))
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep):
            with pytest.raises(httpx.HTTPStatusError):
                await instance._fetch_with_retry(SubQuery(query="q"))
        expected = [3.0**i for i in range(config.max_retries)]
        actual = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual == pytest.approx(expected)

    async def test_max_concurrent_one_serializes_calls(self, config):
        agent_cfg = AgentConfig(max_concurrent=1)
        cls = _make_concrete("fwr_conc")
        order: list[str] = []

        async def slow_search(sq: SubQuery) -> list[RawChunk]:
            order.append("start")
            await asyncio.sleep(0.01)
            order.append("end")
            return []

        instance1 = cls(config, agent_cfg)
        instance2 = cls(config, agent_cfg)
        instance1._search = slow_search
        instance2._search = slow_search

        await asyncio.gather(
            instance1._fetch_with_retry(SubQuery(query="q1")),
            instance2._fetch_with_retry(SubQuery(query="q2")),
        )
        assert order == ["start", "end", "start", "end"]

    async def test_max_concurrent_zero_does_not_restrict(self, config):
        agent_cfg = AgentConfig(max_concurrent=0)
        cls = _make_concrete("fwr_no_conc")
        order: list[str] = []

        async def fast_search(sq: SubQuery) -> list[RawChunk]:
            order.append("start")
            await asyncio.sleep(0)
            order.append("end")
            return []

        instance1 = cls(config, agent_cfg)
        instance2 = cls(config, agent_cfg)
        instance1._search = fast_search
        instance2._search = fast_search

        await asyncio.gather(
            instance1._fetch_with_retry(SubQuery(query="q1")),
            instance2._fetch_with_retry(SubQuery(query="q2")),
        )
        # Both start before either ends when there is no semaphore
        assert order == ["start", "start", "end", "end"]


# ---------------------------------------------------------------------------
# _acquire_rate_limit_slot() / _get_lock() / _reset_rate_limit_state()
# ---------------------------------------------------------------------------


class TestAcquireRateLimitSlot:
    async def test_no_sleep_when_interval_zero(self, config):
        instance = _make_concrete("rl_zero")(config, AgentConfig())
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep):
            await instance._acquire_rate_limit_slot()
        mock_sleep.assert_not_called()

    async def test_no_sleep_on_first_call_with_positive_interval(self, config):
        instance = _make_concrete("rl_first")(config, AgentConfig())
        instance._get_rate_limit_interval = lambda: 1.0
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep), \
             patch("mara.agents.base.time.monotonic", return_value=1000.0):
            await instance._acquire_rate_limit_slot()
        mock_sleep.assert_not_called()

    async def test_sleep_called_when_within_interval(self, config):
        instance = _make_concrete("rl_sleep")(config, AgentConfig())
        instance._get_rate_limit_interval = lambda: 2.0
        agent_type = instance._agent_type()
        SpecialistAgent._last_called[agent_type] = 99.5

        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep), \
             patch("mara.agents.base.time.monotonic", return_value=100.0):
            await instance._acquire_rate_limit_slot()
        mock_sleep.assert_awaited_once_with(pytest.approx(1.5))

    def test_get_lock_creates_asyncio_lock(self, config):
        lock = SpecialistAgent._get_lock("lock_agent")
        assert isinstance(lock, asyncio.Lock)

    def test_get_lock_same_type_returns_same_instance(self, config):
        lock_a = SpecialistAgent._get_lock("same_type")
        lock_b = SpecialistAgent._get_lock("same_type")
        assert lock_a is lock_b

    def test_get_lock_different_types_different_locks(self, config):
        lock_x = SpecialistAgent._get_lock("type_x")
        lock_y = SpecialistAgent._get_lock("type_y")
        assert lock_x is not lock_y

    def test_reset_clears_locks(self, config):
        SpecialistAgent._get_lock("reset_lock_agent")
        assert "reset_lock_agent" in SpecialistAgent._locks
        SpecialistAgent._reset_rate_limit_state()
        assert SpecialistAgent._locks == {}

    def test_reset_clears_last_called(self, config):
        SpecialistAgent._last_called["reset_last_agent"] = 42.0
        SpecialistAgent._reset_rate_limit_state()
        assert SpecialistAgent._last_called == {}

    async def test_zero_division_in_interval_skips_rate_limit(self, config):
        instance = _make_concrete("rl_zdiv")(config, AgentConfig())
        instance._get_rate_limit_interval = MagicMock(side_effect=ZeroDivisionError)
        mock_sleep = AsyncMock()
        with patch("mara.agents.base.asyncio.sleep", mock_sleep):
            await instance._acquire_rate_limit_slot()
        mock_sleep.assert_not_called()

    def test_get_semaphore_creates_asyncio_semaphore(self, config):
        sem = SpecialistAgent._get_semaphore("sem_new_agent", 2)
        assert isinstance(sem, asyncio.Semaphore)

    def test_get_semaphore_same_type_returns_same_instance(self, config):
        sem_a = SpecialistAgent._get_semaphore("sem_same_type", 1)
        sem_b = SpecialistAgent._get_semaphore("sem_same_type", 1)
        assert sem_a is sem_b

    def test_reset_clears_semaphores(self, config):
        SpecialistAgent._get_semaphore("sem_reset_agent", 1)
        assert "sem_reset_agent" in SpecialistAgent._semaphores
        SpecialistAgent._reset_rate_limit_state()
        assert SpecialistAgent._semaphores == {}
