"""Tests for mara.agents.cache."""

from __future__ import annotations

import pytest

from mara.agents.cache import InMemoryCache, NoOpCache, SearchCache
from mara.agents.types import RawChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-03-28T00:00:00+00:00"


def _raw(url: str = "https://example.com", text: str = "hello") -> RawChunk:
    return RawChunk(
        url=url,
        text=text,
        retrieved_at=_TS,
        source_type="abstract",
        sub_query="q",
    )


# ---------------------------------------------------------------------------
# SearchCache Protocol
# ---------------------------------------------------------------------------


class TestSearchCacheProtocol:
    def test_noop_satisfies_protocol(self):
        assert isinstance(NoOpCache(), SearchCache)

    def test_inmemory_satisfies_protocol(self):
        assert isinstance(InMemoryCache(), SearchCache)


# ---------------------------------------------------------------------------
# NoOpCache
# ---------------------------------------------------------------------------


class TestNoOpCache:
    async def test_get_always_returns_none(self):
        cache = NoOpCache()
        result = await cache.get("s2", "quantum computing")
        assert result is None

    async def test_set_then_get_still_none(self):
        cache = NoOpCache()
        await cache.set("s2", "quantum computing", [_raw()])
        result = await cache.get("s2", "quantum computing")
        assert result is None

    async def test_get_any_key_returns_none(self):
        cache = NoOpCache()
        assert await cache.get("core", "biology") is None
        assert await cache.get("pubmed", "cancer") is None


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------


class TestInMemoryCache:
    async def test_miss_returns_none(self):
        cache = InMemoryCache()
        assert await cache.get("s2", "missing query") is None

    async def test_set_then_get_returns_chunks(self):
        cache = InMemoryCache()
        chunks = [_raw()]
        await cache.set("s2", "quantum", chunks)
        result = await cache.get("s2", "quantum")
        assert result == chunks

    async def test_same_instance_returned(self):
        cache = InMemoryCache()
        chunks = [_raw()]
        await cache.set("core", "biology", chunks)
        result = await cache.get("core", "biology")
        assert result is chunks

    async def test_different_agent_types_independent(self):
        cache = InMemoryCache()
        chunks_a = [_raw(url="https://a.com")]
        chunks_b = [_raw(url="https://b.com")]
        await cache.set("s2", "query", chunks_a)
        await cache.set("core", "query", chunks_b)
        assert await cache.get("s2", "query") == chunks_a
        assert await cache.get("core", "query") == chunks_b

    async def test_different_queries_independent(self):
        cache = InMemoryCache()
        chunks_x = [_raw(text="result x")]
        chunks_y = [_raw(text="result y")]
        await cache.set("s2", "query x", chunks_x)
        await cache.set("s2", "query y", chunks_y)
        assert await cache.get("s2", "query x") == chunks_x
        assert await cache.get("s2", "query y") == chunks_y

    async def test_overwrite_updates_value(self):
        cache = InMemoryCache()
        old = [_raw(text="old")]
        new = [_raw(text="new")]
        await cache.set("s2", "q", old)
        await cache.set("s2", "q", new)
        assert await cache.get("s2", "q") == new

    async def test_empty_chunks_list_stored_and_returned(self):
        cache = InMemoryCache()
        await cache.set("s2", "empty", [])
        result = await cache.get("s2", "empty")
        assert result == []

    async def test_independent_instances(self):
        cache_a = InMemoryCache()
        cache_b = InMemoryCache()
        await cache_a.set("s2", "q", [_raw()])
        assert await cache_b.get("s2", "q") is None
