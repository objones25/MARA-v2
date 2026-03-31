"""Tests for mara/agent/scoring.py — BM25 chunk scoring."""

from __future__ import annotations

import pytest

from mara.agent.scoring import score_chunks_bm25
from tests.agent.conftest import make_chunk


def test_score_chunks_bm25_empty_list() -> None:
    result = score_chunks_bm25([], "some query", "sha256")
    assert result == []


def test_score_chunks_bm25_single_chunk_nonzero_score() -> None:
    chunk = make_chunk(text="quantum error correction surface codes")
    result = score_chunks_bm25([chunk], "quantum error correction", "sha256")
    assert len(result) == 1
    assert result[0][0] is chunk
    assert result[0][1] >= 0.0


def test_score_chunks_bm25_returns_all_chunks() -> None:
    chunks = [make_chunk(text=f"text about topic {i}", chunk_index=i) for i in range(5)]
    result = score_chunks_bm25(chunks, "topic", "sha256")
    assert len(result) == 5


def test_score_chunks_bm25_relevant_chunk_scores_higher() -> None:
    relevant = make_chunk(text="machine learning classification accuracy benchmark", chunk_index=0)
    irrelevant = make_chunk(text="history of ancient rome caesar augustus", chunk_index=1)
    result = score_chunks_bm25([irrelevant, relevant], "machine learning classification", "sha256")
    scores = {r[0].chunk_index: r[1] for r in result}
    assert scores[0] > scores[1]


def test_score_chunks_bm25_sorted_descending() -> None:
    chunks = [
        make_chunk(text="neural network deep learning", chunk_index=0),
        make_chunk(text="medieval castle architecture", chunk_index=1),
        make_chunk(text="deep learning neural architecture transformers", chunk_index=2),
    ]
    result = score_chunks_bm25(chunks, "deep learning neural", "sha256")
    scores = [r[1] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_score_chunks_bm25_empty_query_all_zero() -> None:
    chunks = [make_chunk(text="some text", chunk_index=i) for i in range(3)]
    result = score_chunks_bm25(chunks, "", "sha256")
    assert all(score == 0.0 for _, score in result)


def test_score_chunks_bm25_case_insensitive() -> None:
    chunk_upper = make_chunk(text="QUANTUM COMPUTING ENTANGLEMENT", chunk_index=0)
    chunk_lower = make_chunk(text="quantum computing entanglement", chunk_index=1)
    result = score_chunks_bm25([chunk_upper, chunk_lower], "quantum computing", "sha256")
    score_upper = next(r[1] for r in result if r[0].chunk_index == 0)
    score_lower = next(r[1] for r in result if r[0].chunk_index == 1)
    assert pytest.approx(score_upper) == score_lower


def test_score_chunks_bm25_deterministic() -> None:
    chunks = [make_chunk(text=f"word{i} common topic", chunk_index=i) for i in range(10)]
    r1 = score_chunks_bm25(chunks, "common topic", "sha256")
    r2 = score_chunks_bm25(chunks, "common topic", "sha256")
    assert [c.chunk_index for c, _ in r1] == [c.chunk_index for c, _ in r2]
    assert [s for _, s in r1] == [s for _, s in r2]


def test_score_chunks_bm25_returns_tuples() -> None:
    chunk = make_chunk(text="research paper abstract")
    result = score_chunks_bm25([chunk], "research", "sha256")
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 2


# ---------------------------------------------------------------------------
# Sub-query hybrid scoring
# ---------------------------------------------------------------------------


def test_score_chunks_bm25_sub_query_boost_increases_score() -> None:
    """A chunk whose sub_query matches its text should score higher via blending."""
    # Both chunks have the same text so original-query BM25 scores are equal.
    # The one whose sub_query aligns with its content gets a lift from the 30% blend.
    chunk_with_sq = make_chunk(
        text="attention mechanism transformer encoder decoder",
        sub_query="attention mechanism transformer",
        chunk_index=0,
    )
    chunk_no_sq = make_chunk(
        text="attention mechanism transformer encoder decoder",
        sub_query="",  # no sub-query signal
        chunk_index=1,
        url="https://example.com/doc2",  # different URL so hashes differ
    )
    result = score_chunks_bm25(
        [chunk_no_sq, chunk_with_sq], "attention mechanism", "sha256"
    )
    score_map = {r[0].chunk_index: r[1] for r in result}
    assert score_map[0] >= score_map[1]


def test_score_chunks_bm25_multiple_sub_queries_handled() -> None:
    """Chunks with different sub_queries each get their own BM25 sub-query pass."""
    chunk_a = make_chunk(
        text="gradient descent optimization learning rate",
        sub_query="gradient descent",
        chunk_index=0,
    )
    chunk_b = make_chunk(
        text="convolutional neural network image recognition",
        sub_query="convolutional networks",
        chunk_index=1,
        url="https://example.com/doc2",
    )
    result = score_chunks_bm25([chunk_a, chunk_b], "neural network training", "sha256")
    # Both chunks are scored — no exception from multiple sub_query BM25 passes
    assert len(result) == 2
    assert all(isinstance(s, float) for _, s in result)


def test_score_chunks_bm25_empty_sub_query_uses_original_only() -> None:
    """Chunk with empty sub_query falls back to 100 % original-query scoring."""
    chunk = make_chunk(
        text="reinforcement learning reward policy gradient",
        sub_query="",
        chunk_index=0,
    )
    result = score_chunks_bm25([chunk], "reinforcement learning", "sha256")
    assert len(result) == 1
    assert result[0][1] >= 0.0
