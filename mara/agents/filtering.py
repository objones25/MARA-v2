"""Chunk filtering strategies for MARA specialist agents.

Build order: imports only from ``mara.agents.types`` — no intra-package circular risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mara.agents.types import RawChunk


@runtime_checkable
class ChunkFilter(Protocol):
    """Strategy interface for post-chunking filtering.

    Implementations receive the full chunk list and the original query string;
    they return the filtered subset.  Marked ``@runtime_checkable`` so pydantic
    can use ``isinstance`` when validating ``ResearchConfig.chunk_filter``.
    """

    def filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]: ...


@dataclass
class CapFilter:
    """Hard-cap filter: at most *max_chunks_per_url* chunks per source URL,
    then at most *max_chunks_per_agent* chunks in total.

    Order is preserved within each URL; the per-URL cap is applied first,
    then the global cap truncates the tail.
    """

    max_chunks_per_url: int = 3
    max_chunks_per_agent: int = 50

    def filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
        url_counts: dict[str, int] = {}
        result: list[RawChunk] = []
        for chunk in chunks:
            count = url_counts.get(chunk.url, 0)
            if count < self.max_chunks_per_url:
                result.append(chunk)
                url_counts[chunk.url] = count + 1
        return result[: self.max_chunks_per_agent]


@dataclass
class EmbeddingFilter:
    """Stub — not yet implemented.

    Ranks chunks by cosine similarity to the query embedding and keeps the
    top-*max_chunks_per_agent* results above *similarity_threshold*.
    """

    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.6
    max_chunks_per_agent: int = 50

    def filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
        raise NotImplementedError("EmbeddingFilter is not yet implemented")
