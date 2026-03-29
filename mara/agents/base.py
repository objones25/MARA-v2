"""Abstract base class for all MARA specialist agents."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

import httpx

from mara.agents.filtering import ChunkFilter
from mara.agents.types import AgentFindings, RawChunk, SubQuery, VerifiedChunk
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree

_log = logging.getLogger(__name__)


class SpecialistAgent(ABC):
    """Abstract base for all MARA specialist agents.

    Subclasses implement only ``_search()``.  The base class provides
    ``_chunk()`` and ``_filter()`` as overridable concrete stages;
    ``_retrieve()`` wires them into the pipeline.  ``run()`` hashes the
    results and assembles the verified ``AgentFindings``.

    Rate limiting and retry logic live here so individual agents never
    reimplement them.  The base class gates each ``_search()`` invocation
    with ``_acquire_rate_limit_slot()`` and wraps it in
    ``_fetch_with_retry()`` which applies exponential back-off for
    transient HTTP errors.

    **Subclass contract**:

    - Override ``_search()`` as a pure HTTP fetch — no locks, no sleep,
      no retry loops.
    - Set ``_rate_limit_interval`` (class variable) for a fixed inter-call
      delay, or override ``_get_rate_limit_interval()`` for config-dependent
      values (e.g. ``1.0 / config.s2_max_rps``).

    Pipeline::

        _search() → _chunk() → _filter()   [inside _retrieve()]
        _retrieve()                          [called by run()]
        run()                               [public entrypoint]
    """

    # Seconds between consecutive calls to _search() for this agent type.
    # 0.0 means no rate limiting.  Subclasses set this at the class level
    # or override _get_rate_limit_interval() for config-dependent values.
    _rate_limit_interval: float = 0.0

    # Class-level state shared across all instances of the same agent type.
    # Keyed by agent type string so distinct agent types don't interfere.
    _locks: dict[str, asyncio.Lock] = {}
    _last_called: dict[str, float] = {}

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------

    def _get_rate_limit_interval(self) -> float:
        """Return the minimum seconds between consecutive ``_search()`` calls.

        Override in subclasses that need config-dependent intervals, e.g.::

            def _get_rate_limit_interval(self) -> float:
                return 1.0 / self.config.s2_max_rps
        """
        return self._rate_limit_interval

    @classmethod
    def _get_lock(cls, agent_type: str) -> asyncio.Lock:
        """Return (creating if absent) the per-agent-type asyncio.Lock."""
        return cls._locks.setdefault(agent_type, asyncio.Lock())

    @classmethod
    def _reset_rate_limit_state(cls) -> None:
        """Clear all rate-limit state.  Call in test teardown fixtures."""
        cls._locks.clear()
        cls._last_called.clear()

    async def _acquire_rate_limit_slot(self) -> None:
        """Block until it is safe to call ``_search()`` again.

        Enforces ``_get_rate_limit_interval()`` seconds between consecutive
        calls for the same agent type.  No-ops when the interval is ≤ 0.
        """
        interval = self._get_rate_limit_interval()
        if interval <= 0.0:
            return
        agent_type = self._agent_type()
        async with self._get_lock(agent_type):
            now = time.monotonic()
            last = self._last_called.get(agent_type, 0.0)
            wait = interval - (now - last)
            if wait > 0:
                _log.debug(
                    "rate limit: sleeping %.3fs",
                    wait,
                    extra={"agent": agent_type},
                )
                await asyncio.sleep(wait)
            self._last_called[agent_type] = time.monotonic()

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _fetch_with_retry(self, sub_query: SubQuery) -> list[RawChunk]:
        """Call ``_search()`` with caching, rate limiting, and exponential back-off retry.

        Cache check runs first; a hit skips rate limiting and the network call.
        Cache population happens on success only.

        Error classification:

        - ``401``/``403`` — permanent auth failure; re-raise immediately.
        - ``404``         — resource not found; return ``[]``.
        - ``429``/``5xx`` — transient; retry with back-off.
        - ``ConnectError``/``TimeoutException`` — transient; retry with back-off.

        After ``config.max_retries`` failed attempts the final exception
        propagates to the caller.
        """
        agent_type = self._agent_type()
        cache = self.config.search_cache

        cached = await cache.get(agent_type, sub_query.query)
        if cached is not None:
            _log.debug(
                "cache hit: %d chunk(s)",
                len(cached),
                extra={"agent": agent_type, "query": sub_query.query},
            )
            return cached

        await self._acquire_rate_limit_slot()
        last_exc: Exception | None = None
        total = self.config.max_retries + 1
        for attempt in range(total):
            _log.debug(
                "attempt %d/%d",
                attempt + 1,
                total,
                extra={"agent": agent_type},
            )
            try:
                result = await self._search(sub_query)
                await cache.set(agent_type, sub_query.query, result)
                return result
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    _log.warning(
                        "auth error %d — aborting",
                        status,
                        extra={"agent": agent_type},
                    )
                    raise
                if status == 404:
                    _log.debug("404 — returning empty", extra={"agent": agent_type})
                    return []
                _log.warning(
                    "HTTP %d on attempt %d — will retry: %s",
                    status,
                    attempt + 1,
                    exc,
                    extra={"agent": agent_type},
                )
                last_exc = exc
            except httpx.ConnectError as exc:
                _log.warning(
                    "connect error on attempt %d — will retry: %s",
                    attempt + 1,
                    exc,
                    extra={"agent": agent_type},
                )
                last_exc = exc
            except httpx.TimeoutException as exc:
                _log.warning(
                    "timeout on attempt %d — will retry: %s",
                    attempt + 1,
                    exc,
                    extra={"agent": agent_type},
                )
                last_exc = exc
            if attempt < self.config.max_retries:
                backoff = self.config.retry_backoff_base**attempt
                _log.debug(
                    "backing off %.1fs before attempt %d",
                    backoff,
                    attempt + 2,
                    extra={"agent": agent_type},
                )
                await asyncio.sleep(backoff)
        _log.error(
            "all %d attempt(s) failed",
            total,
            extra={"agent": agent_type},
        )
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Agent identity
    # ------------------------------------------------------------------

    def _agent_type(self) -> str:
        """Return the registry name for this agent class.

        Late-binds the registry import to avoid circular imports between
        base.py and registry.py.
        """
        from mara.agents.registry import _REGISTRY

        return next(k for k, v in _REGISTRY.items() if v.cls is type(self))

    def model(self) -> str:
        """Return the model to use for this agent.

        Falls back to ``config.default_model`` when no per-agent override is set.
        """
        return self.config.model_overrides.get(self._agent_type(), self.config.default_model)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    @abstractmethod
    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch raw source chunks for *sub_query*.

        Implementations must NOT hash chunks, implement retries, or acquire
        rate-limit locks — all of that is handled by ``_fetch_with_retry()``.
        Returns unsplit chunks; ``_retrieve()`` calls ``_chunk()`` next.
        """

    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        """Split raw chunks into fixed-size character windows with overlap.

        Uses ``config.chunk_size`` (window size) and ``config.chunk_overlap``
        (characters shared between adjacent windows).  Chunks whose text fits
        within one window are returned unchanged.

        Agents with pre-chunked or structured content (e.g. ArXiv section
        extraction) should override this method and return ``raw`` as-is or
        apply domain-aware splitting.

        Returns *raw* unchanged when ``chunk_size <= 0`` or
        ``chunk_overlap >= chunk_size`` (degenerate configuration).
        """
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = size - overlap
        if size <= 0 or step <= 0:
            return list(raw)

        result: list[RawChunk] = []
        for chunk in raw:
            text = chunk.text
            if len(text) <= size:
                result.append(chunk)
                continue
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                result.append(
                    RawChunk(
                        url=chunk.url,
                        text=text[start:end],
                        retrieved_at=chunk.retrieved_at,
                        source_type=chunk.source_type,
                        sub_query=chunk.sub_query,
                    )
                )
                if end >= len(text):
                    break
                start += step
        return result

    def _filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
        """Filter chunks using the configured ``chunk_filter`` strategy."""
        chunk_filter: ChunkFilter = self.config.chunk_filter
        return chunk_filter.filter(chunks, query)

    async def _retrieve(self, sub_query: SubQuery) -> list[RawChunk]:
        """Pipeline: ``_fetch_with_retry()`` → ``_chunk()`` → ``_filter()``."""
        raw = await self._fetch_with_retry(sub_query)
        agent_type = self._agent_type()
        _log.debug(
            "%s: search returned %d chunk(s)",
            agent_type,
            len(raw),
            extra={"agent": agent_type, "query": sub_query.query},
        )

        chunks = self._chunk(raw)
        if len(chunks) != len(raw):
            _log.debug(
                "chunked %d → %d chunk(s)",
                len(raw),
                len(chunks),
                extra={"agent": agent_type},
            )

        filtered = self._filter(chunks, sub_query.query)
        if len(filtered) != len(chunks):
            _log.debug(
                "filtered %d → %d chunk(s)",
                len(chunks),
                len(filtered),
                extra={"agent": agent_type},
            )

        return filtered

    async def run(self, sub_query: SubQuery) -> AgentFindings:
        """Retrieve, hash, and package source chunks into verified AgentFindings.

        Calls ``_retrieve()``, hashes every chunk with the configured algorithm,
        builds a Merkle tree, and returns a self-verified ``AgentFindings``.
        """
        agent_type = self._agent_type()
        _log.debug(
            "run started: query=%r",
            sub_query.query[:80],
            extra={"agent": agent_type},
        )

        raw_chunks = await self._retrieve(sub_query)

        chunks: tuple[VerifiedChunk, ...] = tuple(
            VerifiedChunk(
                hash=hash_chunk(c.url, c.text, c.retrieved_at, self.config.hash_algorithm),
                url=c.url,
                text=c.text,
                retrieved_at=c.retrieved_at,
                source_type=c.source_type,
                sub_query=c.sub_query,
                chunk_index=i,
            )
            for i, c in enumerate(raw_chunks)
        )

        tree = build_merkle_tree([c.hash for c in chunks], self.config.hash_algorithm)

        findings = AgentFindings(
            agent_type=agent_type,
            query=sub_query.query,
            chunks=chunks,
            merkle_root=tree.root,
            merkle_tree=tree,
        )

        _log.debug(
            "run complete: %d chunk(s), merkle_root=%s",
            findings.chunk_count,
            findings.merkle_root[:8] or "(empty)",
            extra={"agent": agent_type},
        )

        return findings
