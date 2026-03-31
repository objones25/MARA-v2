"""BM25 chunk scoring utilities for the MARA pipeline."""

from __future__ import annotations

from rank_bm25 import BM25Plus

from mara.agents.types import VerifiedChunk

# Weight given to the original research query vs the chunk's originating sub-query.
# 70% original query (global relevance) + 30% sub-query (targeted relevance).
_ALPHA = 0.7


def score_chunks_bm25(
    chunks: list[VerifiedChunk],
    query: str,
    algorithm: str,  # noqa: ARG001 — reserved for future hash-aware ranking
) -> list[tuple[VerifiedChunk, float]]:
    """Score *chunks* by BM25 keyword overlap against *query*.

    Tokenises each chunk's text and the query by lowercasing and splitting on
    whitespace.  Returns a list of ``(chunk, score)`` tuples sorted by score
    descending.  All scores are 0.0 when *query* is empty.

    When *query* is non-empty, blends two BM25 signals:

    * **70 %** — score against *query* (the original research question, global
      relevance).
    * **30 %** — score against each chunk's own ``sub_query`` (the targeted
      aspect it was retrieved to answer).

    This hybrid rewards chunks that are both broadly relevant to the research
    topic *and* precisely answer the aspect they were retrieved for.  Chunks
    with no ``sub_query`` string use the original-query score only.
    """
    if not chunks:
        return []

    tokenised_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Plus(tokenised_corpus)

    query_tokens = query.lower().split()
    orig_scores: list[float] = bm25.get_scores(query_tokens).tolist()

    if not query_tokens:
        # No query context — return zero scores (preserves existing behaviour).
        return sorted(zip(chunks, orig_scores), key=lambda t: t[1], reverse=True)

    # Compute sub-query scores: one BM25 call per unique sub_query string.
    unique_sub_queries = {c.sub_query for c in chunks if c.sub_query}
    sq_scores: dict[str, list[float]] = {
        sq: bm25.get_scores(sq.lower().split()).tolist() for sq in unique_sub_queries
    }

    blended: list[float] = [
        _ALPHA * orig_scores[i] + (1.0 - _ALPHA) * sq_scores[c.sub_query][i]
        if c.sub_query in sq_scores
        else orig_scores[i]
        for i, c in enumerate(chunks)
    ]

    return sorted(zip(chunks, blended), key=lambda t: t[1], reverse=True)
