# MARA Merkle Package Documentation

**Last Updated:** 2026-03-29

## Overview

The `mara/merkle/` sub-package is a self-contained cryptographic library that provides deterministic hashing, Merkle tree construction, proof generation/verification, and forest tree assembly. It enables MARA to produce cryptographically verifiable research reports where any chunk can be verified against the final report's root without re-running the entire pipeline.

**Design Goal:** Standalone sub-package with no intra-package imports. All functions and dataclasses operate on primitive types (strings, lists, dicts) so they remain composable and testable in isolation.

**Core Invariant:** `AgentFindings.__post_init__` recomputes the Merkle root from its chunks. Successful construction = verified. There is no external verification step.

## Table of Contents

1. [Package Structure](#package-structure)
2. [Module: `hasher.py`](#module-hasherpy)
3. [Module: `tree.py`](#module-treepy)
4. [Module: `proof.py`](#module-proofpy)
5. [Module: `forest.py`](#module-forestpy)
6. [Two-Level Proof Walkthrough](#two-level-proof-walkthrough)
7. [Algorithm Notes](#algorithm-notes)
8. [Integration with Pipeline](#integration-with-pipeline)

---

## Package Structure

```text
mara/merkle/
├── __init__.py           # Public API exports
├── hasher.py             # Deterministic serialisation & hashing
├── tree.py               # Merkle tree construction
├── proof.py              # Proof generation & verification
└── forest.py             # Meta-tree over agent sub-trees
```

### Public API (`__init__.py`)

Exports:

- `ForestTree`, `MerkleTree`, `ProofStep` (dataclasses)
- `build_forest_tree()`, `build_merkle_tree()`, `combine_hashes()` (functions)
- `canonical_serialise()`, `hash_chunk()` (hashing)
- `generate_merkle_proof()`, `verify_merkle_proof()` (proofs)

---

## Module: `hasher.py`

### Purpose

Produces deterministic (byte-identical) hashes of source chunks across Python versions, platforms, and locales. The same `(url, text, retrieved_at)` tuple always produces the same hash, enabling independent verification.

### Function: `canonical_serialise(url: str, text: str, retrieved_at: str) -> bytes`

Serialises a chunk's metadata into a deterministic UTF-8 byte string.

**Parameters:**

- `url` — Source URL
- `text` — Chunk text content
- `retrieved_at` — ISO-8601 timestamp string

**Returns:**
Deterministic byte string (UTF-8 encoded JSON)

**Determinism Guarantees:**

```python
# Key settings
json.dumps(
    payload,
    sort_keys=True,           # Key order stable regardless of insertion
    separators=(",", ":"),    # No whitespace variation
    ensure_ascii=True,        # Byte-identical across locales
)
```

- `sort_keys=True` — Dictionary keys are ordered alphabetically, regardless of insertion order or Python version
- `separators=(",", ":")` — No spaces after colons or commas (minimal, deterministic output)
- `ensure_ascii=True` — All non-ASCII characters escaped as `\uXXXX`, ensuring identical bytes on any platform/locale

**Example:**

```python
from mara.merkle import canonical_serialise

url = "https://example.com/paper.pdf"
text = "The quick brown fox jumps over the lazy dog"
retrieved_at = "2026-03-29T12:34:56Z"

data = canonical_serialise(url, text, retrieved_at)
# b'{"retrieved_at":"2026-03-29T12:34:56Z","text":"The quick brown fox...","url":"https://example.com/paper.pdf"}'
```

### Function: `hash_chunk(url: str, text: str, retrieved_at: str, algorithm: str) -> str`

Returns the hex digest of a source chunk using the specified hash algorithm.

**Parameters:**

- `url` — Source URL
- `text` — Chunk text content
- `retrieved_at` — ISO-8601 timestamp string
- `algorithm` — hashlib algorithm name (e.g., `"sha256"`, `"sha512"`). Must match the algorithm used in tree construction.

**Returns:**
Hex digest string (lowercase hexadecimal)

**Algorithm Flow:**

```python
data = canonical_serialise(url, text, retrieved_at)  # Deterministic bytes
h = hashlib.new(algorithm)
h.update(data)
return h.hexdigest()  # Hex string
```

**Default Algorithm:**
The default is `"sha256"` (configured in `ResearchConfig.hash_algorithm`).

**Example:**

```python
from mara.merkle import hash_chunk

hash_value = hash_chunk(
    url="https://example.com/paper.pdf",
    text="Introduction paragraph...",
    retrieved_at="2026-03-29T12:34:56Z",
    algorithm="sha256"
)
# "a3f5b2c1e4d9f8a6b7c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f"
```

---

## Module: `tree.py`

### Tree Construction

Constructs a balanced binary Merkle tree from leaf hashes. Internal nodes combine parent hashes deterministically. If the number of leaves is odd, the last leaf is duplicated to balance the tree.

### Function: `combine_hashes(left: str, right: str, algorithm: str) -> str`

Hashes two hex-digest strings into a parent node hex digest.

**Parameters:**

- `left` — Left child hex digest
- `right` — Right child hex digest
- `algorithm` — hashlib algorithm name (must match leaf hashing algorithm)

**Returns:**
Hex digest of `hash(left_bytes + right_bytes)`

**Algorithm:**

```python
data = (left + right).encode("utf-8")  # Concatenate and encode
h = hashlib.new(algorithm)
h.update(data)
return h.hexdigest()
```

**Note:** Concatenation order matters: `combine_hashes("abc", "def", "sha256")` differs from `combine_hashes("def", "abc", "sha256")`.

**Example:**

```python
from mara.merkle import combine_hashes

left = "a3f5b2c1e4d9f8a6b7c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f"
right = "b4c6d3e2f5a0b9c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3g4h"

parent = combine_hashes(left, right, "sha256")
# "c5d7e4f3g6b1c0d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3g4h5i6"
```

### Dataclass: `MerkleTree`

Represents a balanced binary Merkle tree.

**Attributes:**

```python
@dataclass
class MerkleTree:
    leaves: list[str] = field(default_factory=list)
    levels: list[list[str]] = field(default_factory=list)
    root: str = ""
    algorithm: str = "sha256"
```

- **`leaves`** — Original leaf hashes in insertion order
- **`levels`** — All tree levels; `levels[0]` = leaf level, `levels[-1]` = `[root]`
- **`root`** — Root hash (single element); empty string `""` if the tree is empty
- **`algorithm`** — Hash algorithm used for all internal nodes

**Immutability:**
`MerkleTree` is a mutable dataclass (not frozen), but tree construction is deterministic and stable once built.

### Function: `build_merkle_tree(leaf_hashes: list[str], algorithm: str) -> MerkleTree`

Constructs a `MerkleTree` from a list of hex-digest leaf hashes.

**Parameters:**

- `leaf_hashes` — Ordered list of hex digests (one per source chunk)
- `algorithm` — hashlib algorithm name for internal node hashing. Must match the algorithm used to produce `leaf_hashes`.

**Returns:**
A `MerkleTree` with all levels populated and root set.

**Raises:**

- `ValueError` — If any leaf hash is an empty string

**Balancing Strategy:**

1. Start with leaf-level hashes
2. While more than one element remains:
   - If the current level has an odd number of nodes, duplicate the last node
   - Pair adjacent nodes and hash each pair with `combine_hashes()`
   - Move to the next level
3. The final single node is the root

#### Example: Three Leaves

```text
Input:  ["hash_A", "hash_B", "hash_C"]

Level 0 (leaves):
  [hash_A, hash_B, hash_C, hash_C]  (C duplicated to balance)

Level 1:
  [combine(A, B), combine(C, C)]

Level 2 (root):
  [combine(combine(A, B), combine(C, C))]
```

**Code Example:**

```python
from mara.merkle import build_merkle_tree

leaf_hashes = [
    "a3f5b2c1e4d9f8a6b7c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f",
    "b4c6d3e2f5a0b9c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3g4h",
    "c5d7e4f3g6b1c0d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3g4h5i6",
]

tree = build_merkle_tree(leaf_hashes, algorithm="sha256")

print(tree.root)      # Root hash
print(len(tree.leaves))  # 3
print(len(tree.levels))  # 3 (leaf, intermediate, root)
```

**Edge Cases:**

- **Empty input** — Returns `MerkleTree(algorithm=algorithm)` with empty root `""`
- **Single leaf** — Returns a tree with one level; the leaf is also the root
- **Duplicate leaves** — Allowed; duplicates appear in the tree as-is

---

## Module: `proof.py`

### Proof Mechanism

Generates and verifies Merkle proofs. A proof for leaf `i` is a list of sibling hashes along the path from the leaf to the root. A verifier can recompute the root from the leaf hash and the proof, then check it against the expected root.

### Dataclass: `ProofStep`

Represents one step in a Merkle proof path.

**Attributes:**

```python
@dataclass
class ProofStep:
    sibling_hash: str       # Hash of the sibling node at this level
    position: str           # 'left' | 'right'
```

- **`sibling_hash`** — Hash of the node adjacent to the current node at this level
- **`position`** — Whether the sibling is to the left (`"left"`) or right (`"right"`) of the current node

The position indicates how to combine hashes during verification: if the sibling is to the right, compute `combine_hashes(current, sibling)`. If to the left, compute `combine_hashes(sibling, current)`.

### Function: `generate_merkle_proof(tree: MerkleTree, leaf_index: int) -> list[ProofStep]`

Generates a Merkle proof for the leaf at `leaf_index`.

**Parameters:**

- `tree` — A fully constructed `MerkleTree`
- `leaf_index` — Zero-based index into `tree.leaves`

**Returns:**
Ordered list of `ProofStep` objects from leaf level up to (but not including) the root level.

**Raises:**

- `ValueError` — If the tree is empty
- `IndexError` — If `leaf_index` is out of range

**Algorithm:**

For each level of the tree (except the root):

1. Pad the level if it has an odd number of nodes (same way `build_merkle_tree` does)
2. If the current index is even (left child), the sibling is at `current_index + 1` (right)
3. If the current index is odd (right child), the sibling is at `current_index - 1` (left)
4. Append a `ProofStep` with the sibling hash and position
5. Divide the current index by 2 to move up one level

#### Example: Proof for Leaf 1

```text
Tree structure (3 leaves, C duplicated):
                    root
                   /    \
                  /      \
             combine(A,B) combine(C,C)
              /    \       /    \
            hash_A hash_B hash_C hash_C
            [0]    [1]    [2]    [3]

Proof for leaf_index=1 (hash_B):
  Step 0: sibling_hash=hash_A, position="left"  (B is to the right of A)
  Step 1: sibling_hash=combine(C,C), position="right"
```

**Code Example:**

```python
from mara.merkle import build_merkle_tree, generate_merkle_proof

tree = build_merkle_tree([hash_A, hash_B, hash_C], algorithm="sha256")
proof = generate_merkle_proof(tree, leaf_index=1)

for step in proof:
    print(f"{step.position}: {step.sibling_hash[:8]}...")
```

### Function: `verify_merkle_proof(leaf_hash: str, proof: list[ProofStep], expected_root: str, algorithm: str) -> bool`

Verifies that `leaf_hash` is committed to `expected_root` via the given proof.

**Parameters:**

- `leaf_hash` — Hex digest of the leaf being verified
- `proof` — Proof path as returned by `generate_merkle_proof()`
- `expected_root` — The root hash embedded in the `CertifiedReport`
- `algorithm` — hashlib algorithm name (must match the one used when the tree was built)

**Returns:**
`True` if the recomputed root matches `expected_root`, `False` otherwise.

**Algorithm:**

1. Start with `current = leaf_hash`
2. For each `ProofStep` in the proof (from leaf to root):
   - If `position == "right"`, compute `current = combine_hashes(current, sibling_hash, algorithm)`
   - If `position == "left"`, compute `current = combine_hashes(sibling_hash, current, algorithm)`
3. Return `current == expected_root`

**Code Example:**

```python
from mara.merkle import (
    build_merkle_tree,
    generate_merkle_proof,
    verify_merkle_proof,
)

# Build tree from leaf hashes
tree = build_merkle_tree([hash_A, hash_B, hash_C], algorithm="sha256")

# Generate proof for leaf 1
proof = generate_merkle_proof(tree, leaf_index=1)

# Verify
is_valid = verify_merkle_proof(hash_B, proof, tree.root, algorithm="sha256")
assert is_valid  # True
```

---

## Module: `forest.py`

### Forest Design

Builds a meta-tree over specialist-agent sub-tree roots. Each agent contributes one sub-tree root, which becomes a leaf of the meta-tree. This enables two-level proofs: one level for the chunk within an agent, another for the agent within the forest.

**Design Philosophy:**
`forest.py` is intentionally free of knowledge about agents, configs, or LangGraph. It accepts plain string tuples so the merkle layer remains a self-contained cryptographic library. The corpus assembler is responsible for extracting `(agent_type, merkle_root)` pairs from `AgentFindings`.

### Dataclass: `ForestTree`

Meta-tree whose leaves are specialist-agent sub-tree roots.

**Attributes:**

```python
@dataclass
class ForestTree:
    agent_roots: dict[str, str] = field(default_factory=dict)
    meta_tree: MerkleTree = field(default_factory=MerkleTree)
    root: str = ""
    algorithm: str = "sha256"
```

- **`agent_roots`** — Dictionary mapping `agent_type` (string) to its sub-tree merkle root (hex digest), in alphabetical key order
- **`meta_tree`** — The `MerkleTree` built from the sorted agent roots (treated as leaf hashes)
- **`root`** — Overall root hash; convenience alias for `meta_tree.root`
- **`algorithm`** — Hash algorithm used for meta-tree internal nodes (the verifier CLI reads this field when reconstructing proofs)

**Immutability:**
`ForestTree` is a mutable dataclass, but the `agent_roots` dictionary is constructed once during `build_forest_tree()` and is not mutated afterward.

**Why Alphabetical Order?**

The `agent_roots` dict is sorted alphabetically by `agent_type` before being passed to `build_merkle_tree()`. This ensures the meta-tree root is deterministic regardless of the order in which agents finish in the parallel fan-out. Without this, the root would vary depending on execution order, breaking reproducibility.

### Function: `build_forest_tree(agent_data: Sequence[tuple[str, str]], algorithm: str) -> ForestTree`

Builds a `ForestTree` from a sequence of `(agent_type, merkle_root)` pairs.

**Parameters:**

- `agent_data` — Sequence of tuples; each tuple is `(agent_type, merkle_root)` where:
  - `agent_type` — String identifier (e.g., `"arxiv"`, `"pubmed"`, `"s2"`)
  - `merkle_root` — Hex digest of that agent's sub-tree root
  - Each `agent_type` must be unique
  - Order does not matter — leaves are sorted alphabetically
- `algorithm` — hashlib algorithm name for meta-tree internal nodes. Must match the algorithm used to produce each sub-tree merkle root.

**Returns:**
A `ForestTree` with:

- `agent_roots` populated (sorted by agent_type)
- `meta_tree` built from the sorted roots
- `root` set to the meta-tree root
- `algorithm` preserved

**Raises:**

- `ValueError` — If `agent_data` contains duplicate `agent_type` values

**Algorithm:**

1. Check for duplicate `agent_type` values; raise if any found
2. Sort `agent_data` alphabetically by `agent_type`
3. Create `agent_roots` dict from sorted pairs
4. Extract root values in alphabetical order as leaf hashes
5. Call `build_merkle_tree(leaf_hashes, algorithm)` to create the meta-tree
6. Return `ForestTree` with all fields populated

**Code Example:**

```python
from mara.merkle import build_forest_tree

agent_data = [
    ("pubmed", "hash_pubmed_root"),
    ("arxiv", "hash_arxiv_root"),
    ("s2", "hash_s2_root"),
]

forest = build_forest_tree(agent_data, algorithm="sha256")

# agent_roots is sorted alphabetically:
# {"arxiv": "hash_arxiv_root", "pubmed": "hash_pubmed_root", "s2": "hash_s2_root"}

print(forest.root)  # Meta-tree root
print(forest.agent_roots)  # Sorted dict
```

**Edge Cases:**

- **Empty input** — Returns `ForestTree(algorithm=algorithm)` with empty root `""`
- **Single agent** — Returns a forest with one level; the agent's root is also the meta-tree root
- **Duplicate agent_type** — Raises `ValueError` with a descriptive message

---

## Two-Level Proof Walkthrough

This example demonstrates how MARA's two-level proof structure enables verification of a single chunk against the final `CertifiedReport`.

### Setup

Suppose we have:

- 3 agents: `arxiv`, `pubmed`, `s2`
- Each agent retrieved multiple chunks

**Agent Sub-Trees:**

```text
arxiv (4 chunks):
  Chunk 0: text="Introduction..."
  Chunk 1: text="Methods..."
  Chunk 2: text="Results..."
  Chunk 3: text="Conclusion..."
  → Sub-tree root: arxiv_root

pubmed (2 chunks):
  Chunk 0: text="Abstract of..."
  Chunk 1: text="Full text of..."
  → Sub-tree root: pubmed_root

s2 (3 chunks):
  Chunk 0: text="Snippet from..."
  Chunk 1: text="Snippet from..."
  Chunk 2: text="Snippet from..."
  → Sub-tree root: s2_root
```

**Meta-Tree (ForestTree):**

```text
Leaf order (alphabetical): [arxiv_root, pubmed_root, s2_root]

                      forest_root
                     /           \
                    /             \
              combine(arxiv_root,  combine(pubmed_root,
              pubmed_root)         s2_root)
              /           \        /                \
        arxiv_root    pubmed_root s2_root      s2_root
```

### Verification Example: arxiv Chunk 2

Goal: Verify that the chunk "Results..." retrieved from arxiv is committed to the `CertifiedReport.forest_tree.root`.

#### Step 1: Chunk-Level Proof (within the arxiv sub-tree)

```python
from mara.merkle import generate_merkle_proof

# arxiv's sub-tree was built from 4 leaves
arxiv_tree = MerkleTree(
    leaves=[hash_0, hash_1, hash_2, hash_3],
    levels=[
        [hash_0, hash_1, hash_2, hash_3, hash_3],  # leaf + padded
        [combine(hash_0, hash_1), combine(hash_2, hash_3)],
        [arxiv_root]
    ],
    root=arxiv_root,
    algorithm="sha256"
)

# Generate proof for chunk 2
chunk_proof = generate_merkle_proof(arxiv_tree, leaf_index=2)
# Output:
#   ProofStep(sibling_hash=combine(hash_2, hash_3), position="right")
#   ProofStep(sibling_hash=combine(hash_0, hash_1), position="left")
```

#### Step 2: Agent-Level Proof (arxiv's place in the forest)

```python
from mara.merkle import generate_merkle_proof

# The forest meta-tree has leaves [arxiv_root, pubmed_root, s2_root]
# We need a proof that arxiv_root is committed to forest_root

agent_proof = generate_merkle_proof(forest_tree.meta_tree, leaf_index=0)
# Output (arxiv is at leaf_index=0 in alphabetical order):
#   ProofStep(sibling_hash=combine(pubmed_root, s2_root), position="right")
```

#### Step 3: Full Verification

```python
from mara.merkle import verify_merkle_proof, hash_chunk

# Recompute chunk hash (or it comes from the VerifiedChunk)
chunk_hash = hash_chunk(
    url="https://arxiv.org/pdf/...",
    text="Results...",
    retrieved_at="2026-03-29T12:34:56Z",
    algorithm="sha256"
)

# Verify chunk is committed to arxiv_root
is_chunk_valid = verify_merkle_proof(
    chunk_hash,
    chunk_proof,
    arxiv_root,  # Agent's sub-tree root
    algorithm="sha256"
)
assert is_chunk_valid  # True

# Verify arxiv_root is committed to forest_root
is_agent_valid = verify_merkle_proof(
    arxiv_root,
    agent_proof,
    forest_tree.root,  # Overall forest root
    algorithm="sha256"
)
assert is_agent_valid  # True
```

**Result:**
Both proofs verify successfully, confirming that the chunk "Results..." is part of the research report's verified corpus.

---

## Algorithm Notes

### Default Hash Algorithm

The default hash algorithm is **SHA-256** (`"sha256"`), configured in `ResearchConfig.hash_algorithm`.

All functions that accept an `algorithm` parameter default to or respect the config value:

- `hash_chunk(url, text, retrieved_at, algorithm)` — required parameter
- `combine_hashes(left, right, algorithm)` — required parameter
- `build_merkle_tree(leaf_hashes, algorithm)` — required parameter
- `build_forest_tree(agent_data, algorithm)` — required parameter

### Changing the Algorithm

To use a different algorithm (e.g., SHA-512):

```python
from mara.merkle import build_merkle_tree, hash_chunk

leaf_hashes = [
    hash_chunk(url, text, retrieved_at, algorithm="sha512"),
    # ... more chunks
]

tree = build_merkle_tree(leaf_hashes, algorithm="sha512")
```

Ensure all hashing operations use the same algorithm throughout the pipeline.

### Determinism Guarantees

The `canonical_serialise()` function guarantees byte-identical output across:

- **Python versions** — JSON with `sort_keys=True` and fixed separators
- **Platforms** — `ensure_ascii=True` escapes non-ASCII characters
- **Locales** — No locale-dependent operations

This allows independent verification: given the same `(url, text, retrieved_at)` tuple, any verifier can recompute the same hash.

### Algorithm Validation

The merkle package does not validate algorithm names; it delegates to `hashlib.new(algorithm)`. Invalid algorithm names will raise `ValueError` from hashlib at runtime.

```python
import hashlib

# Valid
hashlib.new("sha256")   # OK
hashlib.new("sha512")   # OK

# Invalid
hashlib.new("sha999")   # ValueError: unsupported hash type sha999
```

---

## Integration with Pipeline

### Where Merkle Functions Are Called

**`mara/agents/base.py` — `SpecialistAgent.run()`**

After retrieving chunks, each agent hashes them and builds a sub-tree:

```python
async def run(self, sub_query: SubQuery) -> AgentFindings:
    # 1. Retrieve raw chunks
    raw_chunks = await self._retrieve(sub_query)

    # 2. Hash each chunk
    verified_chunks = []
    for i, raw in enumerate(raw_chunks):
        hash_value = hash_chunk(
            url=raw.url,
            text=raw.text,
            retrieved_at=raw.retrieved_at,
            algorithm=self.config.hash_algorithm,
        )
        verified_chunks.append(
            VerifiedChunk(..., hash=hash_value, chunk_index=i)
        )

    # 3. Build sub-tree
    leaf_hashes = [vc.hash for vc in verified_chunks]
    merkle_tree = build_merkle_tree(leaf_hashes, self.config.hash_algorithm)

    # 4. Return findings with merkle_root and merkle_tree
    return AgentFindings(
        agent_type=self.agent_type,
        query=sub_query.query,
        chunks=tuple(verified_chunks),
        merkle_root=merkle_tree.root,
        merkle_tree=merkle_tree,
    )
```

**`mara/agent/nodes/corpus_assembler.py` — Forest Construction**

The corpus assembler gathers sub-tree roots from all agents and builds the forest:

```python
def corpus_assembler_node(state: GraphState, config: ResearchConfig) -> dict:
    findings: list[AgentFindings] = state.get("findings", [])

    # Extract (agent_type, merkle_root) pairs
    agent_data = [
        (f.agent_type, f.merkle_root)
        for f in findings
        if f.chunk_count > 0  # Skip agents with 0 chunks
    ]

    # Build forest tree
    forest_tree = build_forest_tree(
        agent_data,
        algorithm=config.hash_algorithm,
    )

    # Assign global chunk indices and build CertifiedReport
    # ...
    return {"forest_tree": forest_tree, "flattened_chunks": ...}
```

### AgentFindings.**post_init** Verification

The `AgentFindings` dataclass re-verifies the merkle root on construction:

```python
@dataclass(frozen=True)
class AgentFindings:
    agent_type: str
    query: str
    chunks: tuple[VerifiedChunk, ...]
    merkle_root: str
    merkle_tree: MerkleTree

    def __post_init__(self):
        # Recompute root from chunks
        leaf_hashes = [vc.hash for vc in self.chunks]
        recomputed_tree = build_merkle_tree(leaf_hashes, self.merkle_tree.algorithm)

        # Check consistency
        if recomputed_tree.root != self.merkle_root:
            raise ValueError("merkle_root mismatch")
```

This invariant ensures that any `AgentFindings` instance is cryptographically verified at construction time. There is no separate verification step.

### CertifiedReport Output

The final `CertifiedReport` includes the `ForestTree`:

```python
@dataclass(frozen=True)
class CertifiedReport:
    original_query: str
    report: str
    forest_tree: ForestTree  # Meta-tree for two-level verification
    chunks: tuple[VerifiedChunk, ...]  # All chunks with their global indices
```

A verifier can use `forest_tree.root` as the public hash to verify any chunk offline:

```python
# Verify a chunk from the report
chunk_hash = chunk.hash  # From CertifiedReport.chunks
chunk_proof = generate_merkle_proof(agent_tree, chunk_index)
agent_proof = generate_merkle_proof(forest_tree.meta_tree, agent_leaf_index)

# Two-level verification
chunk_valid = verify_merkle_proof(
    chunk_hash, chunk_proof, agent_root, algorithm
)
agent_valid = verify_merkle_proof(
    agent_root, agent_proof, forest_tree.root, algorithm
)

assert chunk_valid and agent_valid
```

---

## Summary

The `mara/merkle/` package provides:

1. **Deterministic hashing** — same input always produces the same hash, across platforms and locales
2. **Merkle trees** — balanced binary trees with odd-leaf duplication for stability
3. **Merkle proofs** — sibling-list verification without re-hashing the entire tree
4. **Forest meta-tree** — two-level proof structure enabling offline verification of any chunk

The design is intentionally standalone: no intra-package imports, no dependencies on agents or configs, no knowledge of LangGraph. This keeps the cryptographic layer simple, testable, and reusable.

**Core guarantee:** Any `VerifiedChunk` can be verified against a `CertifiedReport.forest_tree.root` using two independent Merkle proofs (chunk-level + agent-level), without re-running the pipeline.
