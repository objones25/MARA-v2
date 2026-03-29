"""Core data types shared across all MARA agents.

Build order: this module has no intra-package imports — import freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mara.merkle.tree import MerkleTree

if TYPE_CHECKING:
    from mara.merkle.forest import ForestTree


@dataclass
class SubQuery:
    """A single decomposed research sub-question.

    ``domain`` carries an optional hint for the query planner (e.g. ``"empirical"``,
    ``"clinical"``) to guide sub-query decomposition. Defaults to empty string.

    ``query`` is stripped of leading/trailing whitespace on construction. An
    empty or whitespace-only query raises ``ValueError`` immediately.
    """

    query: str
    domain: str = ""
    agent: str = ""  # set by query planner for directed routing; empty = broadcast

    def __post_init__(self) -> None:
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("SubQuery.query must not be empty or whitespace")


@dataclass
class RawChunk:
    """Unverified source chunk returned by an agent's ``_retrieve()`` method.

    ``chunk_index`` is not present here — it is assigned by
    ``SpecialistAgent.run()`` during enumeration. Agents never need to
    pre-assign ordering.

    Empty ``url`` or ``text`` (including whitespace-only) raises ``ValueError``
    immediately to prevent garbage leaves from reaching the hasher.
    """

    url: str
    text: str
    retrieved_at: str  # ISO-8601 UTC
    source_type: str
    sub_query: str

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("RawChunk.url must not be empty")
        if not self.text.strip():
            raise ValueError("RawChunk.text must not be empty")


@dataclass(frozen=True)
class VerifiedChunk:
    """Source chunk whose content has been cryptographically hashed.

    The ``hash`` field is SHA-256 of ``canonical_serialise(url, text, retrieved_at)``.
    Produced exclusively by ``SpecialistAgent.run()`` — never constructed manually.

    ``chunk_index`` is the position within the *agent's own* chunk list for this
    sub-query. It is not a global index across all findings. The corpus assembler
    assigns global indices when flattening all ``AgentFindings`` into one list.
    """

    hash: str
    url: str
    text: str
    retrieved_at: str  # ISO-8601 UTC
    source_type: str
    sub_query: str
    chunk_index: int

    @property
    def short_hash(self) -> str:
        """First 8 hex characters of the hash, for use in citations and logging."""
        return self.hash[:8]


@dataclass(frozen=True)
class AgentFindings:
    """Verified output from one specialist agent for one sub-query.

    Invariant: ``merkle_root == build_merkle_tree([c.hash for c in chunks]).root``.
    Verified in ``__post_init__``; a successfully constructed ``AgentFindings`` is a
    proof that its Merkle tree was built from these exact chunks.
    """

    agent_type: str
    query: str
    chunks: tuple[VerifiedChunk, ...]
    merkle_root: str
    merkle_tree: MerkleTree = field(compare=False)

    def __post_init__(self) -> None:
        from mara.merkle.tree import build_merkle_tree

        expected_root = build_merkle_tree(
            [c.hash for c in self.chunks], self.merkle_tree.algorithm
        ).root
        if self.merkle_root != expected_root:
            raise ValueError(
                f"merkle_root mismatch: provided {self.merkle_root!r}, "
                f"computed {expected_root!r}"
            )

    @property
    def chunk_count(self) -> int:
        """Number of verified chunks in this findings object."""
        return len(self.chunks)


@dataclass(frozen=True)
class CertifiedReport:
    """A cryptographically certified research report.

    Bundles the final narrative with the full provenance chain:
    the ``ForestTree`` (meta-tree of all agent sub-trees) and the
    flattened, globally-indexed list of every ``VerifiedChunk`` used.

    Inline citations in ``report`` use the format ``[ML:index:hash]``
    where ``index`` is the chunk's global ``chunk_index`` and ``hash``
    is its ``short_hash``.
    """

    original_query: str
    report: str
    forest_tree: "ForestTree"
    chunks: tuple[VerifiedChunk, ...]
