"""BM25 chunk scoring utilities for the MARA pipeline."""

from __future__ import annotations

from rank_bm25 import BM25Plus

from mara.agents.types import VerifiedChunk


def score_chunks_bm25(
    chunks: list[VerifiedChunk],
    query: str,
    algorithm: str,  # noqa: ARG001 — reserved for future hash-aware ranking
) -> list[tuple[VerifiedChunk, float]]:
    """Score *chunks* by BM25 keyword overlap against *query*.

    Tokenises each chunk's text and the query by lowercasing and splitting on
    whitespace.  Returns a list of ``(chunk, score)`` tuples sorted by score
    descending.  All scores are 0.0 when *query* is empty.
    """
    if not chunks:
        return []

    tokenised_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Plus(tokenised_corpus)

    query_tokens = query.lower().split()
    scores: list[float] = bm25.get_scores(query_tokens).tolist()

    return sorted(zip(chunks, scores), key=lambda t: t[1], reverse=True)
