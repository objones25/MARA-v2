"""Search result cache strategies for SpecialistAgent.

The ``SearchCache`` Protocol defines the interface.  ``NoOpCache`` is the
default (no caching).  ``InMemoryCache`` stores results in a plain dict for
the lifetime of one pipeline run — useful when sub-queries overlap.

A ``RedisCache`` or any other backend can be added later as a drop-in by
implementing the same two-method Protocol; no base class changes are needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mara.agents.types import RawChunk


@runtime_checkable
class SearchCache(Protocol):
    """Async key-value cache keyed on ``(agent_type, query)``."""

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        """Return cached chunks or ``None`` on a miss."""
        ...

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        """Store *chunks* under ``(agent_type, query)``."""
        ...


class NoOpCache:
    """Default cache — all lookups miss, writes are discarded."""

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        return None

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        pass


class InMemoryCache:
    """Per-pipeline-run dict cache.

    Useful when the query planner emits overlapping sub-queries directed at
    the same agent.  Stored for the lifetime of the ``InMemoryCache`` instance.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[RawChunk]] = {}

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        return self._store.get((agent_type, query))

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        self._store[(agent_type, query)] = chunks
