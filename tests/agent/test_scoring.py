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
