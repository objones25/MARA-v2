"""Abstract base class for all MARA specialist agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mara.agents.types import AgentFindings, RawChunk, SubQuery, VerifiedChunk
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree


class SpecialistAgent(ABC):
    """Abstract base for all MARA specialist agents.

    Subclasses implement only ``_retrieve()``. The base ``run()`` method
    handles hashing, Merkle tree construction, and AgentFindings assembly.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

    def _agent_type(self) -> str:
        """Return the registry name for this agent class.

        Late-binds the registry import to avoid circular imports between
        base.py and registry.py.
        """
        from mara.agents.registry import _REGISTRY

        return next(k for k, v in _REGISTRY.items() if v is type(self))

    def model(self) -> str:
        """Return the model to use for this agent.

        Falls back to ``config.default_model`` when no per-agent override is set.
        """
        return self.config.model_overrides.get(self._agent_type(), self.config.default_model)

    @abstractmethod
    async def _retrieve(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch raw source chunks for *sub_query*.

        Implementations must NOT hash chunks — hashing is done by ``run()``.
        The ``chunk_index`` field on returned ``RawChunk`` objects is ignored;
        ``run()`` assigns sequential indices.
        """

    async def run(self, sub_query: SubQuery) -> AgentFindings:
        """Retrieve, hash, and package source chunks into verified AgentFindings.

        Calls ``_retrieve()``, hashes every chunk with the configured algorithm,
        builds a Merkle tree, and returns a self-verified ``AgentFindings``.
        """
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

        return AgentFindings(
            agent_type=self._agent_type(),
            query=sub_query.query,
            chunks=chunks,
            merkle_root=tree.root,
            merkle_tree=tree,
        )
