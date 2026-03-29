"""Tests for mara/agents/filtering.py — ChunkFilter, CapFilter, EmbeddingFilter."""

import pytest

from mara.agents.filtering import CapFilter, ChunkFilter, EmbeddingFilter
from mara.agents.types import RawChunk

_TS = "2026-03-28T00:00:00+00:00"


def _chunk(
    url: str = "https://example.com",
    text: str = "content",
    sub_query: str = "q",
) -> RawChunk:
    return RawChunk(url=url, text=text, retrieved_at=_TS, source_type="abstract", sub_query=sub_query)


# ---------------------------------------------------------------------------
# ChunkFilter protocol
# ---------------------------------------------------------------------------


class TestChunkFilterProtocol:
    def test_cap_filter_satisfies_protocol(self):
        assert isinstance(CapFilter(), ChunkFilter)

    def test_embedding_filter_satisfies_protocol(self):
        assert isinstance(EmbeddingFilter(), ChunkFilter)


# ---------------------------------------------------------------------------
# CapFilter
# ---------------------------------------------------------------------------


class TestCapFilter:
    def test_empty_chunks_returns_empty(self):
        assert CapFilter().filter([], "q") == []

    def test_passes_chunks_within_limits(self):
        chunks = [_chunk(url=f"https://example.com/{i}") for i in range(3)]
        result = CapFilter(max_chunks_per_url=5, max_chunks_per_agent=50).filter(chunks, "q")
        assert result == chunks

    def test_caps_per_url(self):
        chunks = [_chunk(url="https://same.com", text=f"text {i}") for i in range(5)]
        result = CapFilter(max_chunks_per_url=3).filter(chunks, "q")
        assert len(result) == 3
        assert all(c.url == "https://same.com" for c in result)

    def test_caps_total_chunks(self):
        chunks = [_chunk(url=f"https://example.com/{i}") for i in range(10)]
        result = CapFilter(max_chunks_per_url=10, max_chunks_per_agent=5).filter(chunks, "q")
        assert len(result) == 5

    def test_per_url_cap_applied_before_total_cap(self):
        # 2 URLs × 4 chunks each = 8 raw; per-url cap=2 → 4; total cap=3 → 3
        chunks = (
            [_chunk(url="https://a.com", text=f"a{i}") for i in range(4)]
            + [_chunk(url="https://b.com", text=f"b{i}") for i in range(4)]
        )
        result = CapFilter(max_chunks_per_url=2, max_chunks_per_agent=3).filter(chunks, "q")
        assert len(result) == 3

    def test_query_is_ignored(self):
        chunks = [_chunk()]
        assert CapFilter().filter(chunks, "irrelevant query") == chunks

    def test_preserves_insertion_order_within_url(self):
        chunks = [_chunk(url="https://a.com", text=f"text {i}") for i in range(3)]
        result = CapFilter(max_chunks_per_url=3).filter(chunks, "q")
        assert [c.text for c in result] == ["text 0", "text 1", "text 2"]

    def test_multiple_urls_each_independently_capped(self):
        chunks = (
            [_chunk(url="https://a.com", text=f"a{i}") for i in range(4)]
            + [_chunk(url="https://b.com", text=f"b{i}") for i in range(4)]
        )
        result = CapFilter(max_chunks_per_url=2, max_chunks_per_agent=100).filter(chunks, "q")
        a_chunks = [c for c in result if c.url == "https://a.com"]
        b_chunks = [c for c in result if c.url == "https://b.com"]
        assert len(a_chunks) == 2
        assert len(b_chunks) == 2

    def test_default_fields(self):
        f = CapFilter()
        assert f.max_chunks_per_url == 3
        assert f.max_chunks_per_agent == 50


# ---------------------------------------------------------------------------
# EmbeddingFilter
# ---------------------------------------------------------------------------


class TestEmbeddingFilter:
    def test_filter_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="EmbeddingFilter is not yet implemented"):
            EmbeddingFilter().filter([], "q")

    def test_filter_raises_with_chunks(self):
        with pytest.raises(NotImplementedError):
            EmbeddingFilter().filter([_chunk()], "some query")

    def test_default_model_field(self):
        assert EmbeddingFilter().model == "all-MiniLM-L6-v2"

    def test_default_similarity_threshold(self):
        assert EmbeddingFilter().similarity_threshold == pytest.approx(0.6)

    def test_default_max_chunks_per_agent(self):
        assert EmbeddingFilter().max_chunks_per_agent == 50

    def test_fields_can_be_overridden(self):
        f = EmbeddingFilter(model="custom-model", similarity_threshold=0.8, max_chunks_per_agent=10)
        assert f.model == "custom-model"
        assert f.similarity_threshold == pytest.approx(0.8)
        assert f.max_chunks_per_agent == 10
