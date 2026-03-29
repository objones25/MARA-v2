"""Abstract base class for all MARA specialist agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

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

    Pipeline::

        _search() → _chunk() → _filter()   [inside _retrieve()]
        _retrieve()                          [called by run()]
        run()                               [public entrypoint]
    """

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

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

    @abstractmethod
    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch raw source chunks for *sub_query*.

        Implementations must NOT hash chunks — hashing is done by ``run()``.
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
        """Pipeline: ``_search()`` → ``_chunk()`` → ``_filter()``."""
        raw = await self._search(sub_query)
        _log.debug(
            "search returned %d chunk(s)",
            len(raw),
            extra={"agent": self._agent_type(), "query": sub_query.query},
        )

        chunks = self._chunk(raw)
        if len(chunks) != len(raw):
            _log.debug(
                "chunked %d → %d chunk(s)",
                len(raw),
                len(chunks),
                extra={"agent": self._agent_type()},
            )

        filtered = self._filter(chunks, sub_query.query)
        if len(filtered) != len(chunks):
            _log.debug(
                "filtered %d → %d chunk(s)",
                len(chunks),
                len(filtered),
                extra={"agent": self._agent_type()},
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
