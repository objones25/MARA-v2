"""Merkle forest: meta-tree over specialist-agent sub-trees.

Each agent's sub-tree root becomes a leaf of a single meta-tree. Leaf order
is alphabetical by agent_type so the root is deterministic regardless of
agent execution order (which is non-deterministic in a parallel fan-out).

``build_forest_tree`` is intentionally free of knowledge about agents,
configs, or LangGraph. It accepts plain string tuples so the merkle layer
remains a self-contained cryptographic library. The corpus assembler is
responsible for extracting (agent_type, merkle_root) pairs from AgentFindings.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from mara.merkle.tree import MerkleTree, build_merkle_tree


@dataclass
class ForestTree:
    """Meta-tree whose leaves are specialist-agent sub-tree roots.

    Attributes:
        agent_roots: Mapping of agent_type -> sub-tree merkle root, in
                     alphabetical key order (the same order used as meta-tree
                     leaf input).
        meta_tree:   The MerkleTree built from the sorted agent roots.
        root:        Overall root hash; convenience alias for meta_tree.root.
        algorithm:   Hash algorithm used for meta-tree internal nodes.
                     The verifier CLI reads this field when reconstructing
                     proofs; do not treat it as optional metadata.
    """

    agent_roots: dict[str, str] = field(default_factory=dict)
    meta_tree: MerkleTree = field(default_factory=MerkleTree)
    root: str = ""
    algorithm: str = "sha256"


def build_forest_tree(
    agent_data: Sequence[tuple[str, str]],
    algorithm: str,
) -> ForestTree:
    """Build a ForestTree from a sequence of (agent_type, merkle_root) pairs.

    Leaf order for the meta-tree is alphabetical by agent_type. This makes
    the meta-tree root deterministic regardless of the order in which agents
    finish in the parallel fan-out.

    Args:
        agent_data: Sequence of (agent_type, merkle_root) pairs, one per
                    specialist agent. Each agent_type must be unique. Order
                    does not matter — leaves are sorted alphabetically.
        algorithm:  hashlib algorithm name for meta-tree internal nodes.
                    Must match the algorithm used to produce each sub-tree
                    merkle_root (i.e. ResearchConfig.hash_algorithm).

    Returns:
        A ForestTree with agent_roots, meta_tree, root, and algorithm set.

    Raises:
        ValueError: If agent_data contains duplicate agent_type values.
    """
    if not agent_data:
        return ForestTree(algorithm=algorithm)

    seen: set[str] = set()
    for agent_type, _ in agent_data:
        if agent_type in seen:
            raise ValueError(
                f"Duplicate agent_type {agent_type!r} in agent_data; "
                "each agent may contribute at most one sub-tree."
            )
        seen.add(agent_type)

    sorted_pairs = sorted(agent_data, key=lambda t: t[0])
    agent_roots = {agent_type: root for agent_type, root in sorted_pairs}
    leaf_hashes = list(agent_roots.values())

    meta_tree = build_merkle_tree(leaf_hashes, algorithm)

    return ForestTree(
        agent_roots=agent_roots,
        meta_tree=meta_tree,
        root=meta_tree.root,
        algorithm=algorithm,
    )
