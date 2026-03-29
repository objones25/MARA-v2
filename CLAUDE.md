# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Run tests with coverage (also enforced automatically via addopts)
uv run pytest --cov=mara --cov-branch --cov-report=term-missing --cov-fail-under=98

# Run the CLI
uv run mara run "your research query"
```

## Architecture

MARA is a multi-agent research pipeline that produces cryptographically verifiable reports. Each agent retrieves raw source text, hashes it immediately, and assembles hashes into a Merkle sub-tree. Sub-tree roots become leaves of a meta-tree (`ForestTree`). Any chunk is verifiable with a two-level proof without re-running MARA.

**Core invariant:** `AgentFindings.__post_init__` recomputes the Merkle root from its chunks. Successful construction = verified. There is no external verification step.

### Data Flow

```
query_planner → Send() × N sub-queries × M agents → corpus_assembler → report_synthesizer → certified_output
```

Fan-out uses LangGraph's `Send()` API. Fan-in uses `operator.add` on the state's `AgentFindings` list.

### `mara/agents/types.py` — Pipeline data types

Four dataclasses form the data contract between pipeline stages:

- **`SubQuery`** — mutable dataclass: `query: str`, `domain: str = ""`. `query` is stripped of whitespace on construction; empty/whitespace-only raises `ValueError`. `domain` is an optional hint for the query planner (e.g. `"empirical"`, `"clinical"`).
- **`RawChunk`** — mutable dataclass: `url`, `text`, `retrieved_at`, `source_type`, `sub_query`. Returned by `_retrieve()`. No `chunk_index` — that is assigned by `run()` during enumeration. Empty/whitespace `url` or `text` raises `ValueError` at construction.
- **`VerifiedChunk`** — frozen dataclass: all `RawChunk` fields plus `hash` (SHA-256 of `canonical_serialise(url, text, retrieved_at)`) and `chunk_index` (position within the agent's own chunk list — not a global index; the corpus assembler assigns global indices when flattening). `short_hash` property returns the first 8 hex characters for citations and logging. Produced exclusively by `SpecialistAgent.run()`.
- **`AgentFindings`** — frozen dataclass: `agent_type`, `query`, `chunks: tuple[VerifiedChunk, ...]`, `merkle_root`, `merkle_tree`. `__post_init__` recomputes the Merkle root from `chunks` and raises `ValueError("merkle_root mismatch")` if it doesn't match. `chunk_count` property returns `len(chunks)`. A successfully constructed `AgentFindings` is already verified — no external step needed.

### `mara/agents/registry.py` — Self-registration

`_REGISTRY: dict[str, type[SpecialistAgent]]` is the single source of truth for all agents.

- `@agent("name")` — class decorator that inserts the class into `_REGISTRY`. Raises `ValueError` if the name is already taken (no silent overwrites).
- `get_agents(config)` — instantiates every registered class with `config` and returns the list. Called by the graph to build the agent pool at runtime.

Adding a new agent requires only writing the class and applying `@agent("name")` — nothing else in the pipeline changes.

### `mara/agents/base.py` — SpecialistAgent

Abstract base class all agents extend. Subclasses implement only `_retrieve(sub_query) -> list[RawChunk]`.

- **`__init__(self, config)`** — stores `ResearchConfig` on `self.config`.
- **`_agent_type()`** — reverse-looks up `type(self)` in `_REGISTRY` to return the registered name. Import is deferred inside the method body to avoid a circular import between `base` and `registry`.
- **`model()`** — returns `config.model_overrides.get(agent_type, config.default_model)`. Allows per-agent-type model selection without subclass boilerplate.
- **`run(sub_query)`** — the only public method. Calls `_retrieve()`, hashes each `RawChunk` with `config.hash_algorithm`, assigns sequential `chunk_index` values, builds a `MerkleTree`, and returns a self-verified `AgentFindings`. Subclasses never touch hashing or tree construction.

Each agent module defines its own `source_type` string constants (e.g., `LATEX`, `ABSTRACT_ONLY`).

#### Agent Content Strategies

| Agent | Discovery | Content |
|---|---|---|
| **arxiv** | `export.arxiv.org/api/query` (Atom XML, stdlib) | `.tar.gz` → LaTeX → PDF in tarball → rendered PDF → abstract |
| **s2** | `api.semanticscholar.org/graph/v1/snippet/search` | Snippets inline; rate-limited to ≤1 RPS via `asyncio.Lock` |
| **pubmed** | NCBI `esearch` + `esummary` | PMC XML (`<sec>`) → abstract |
| **core** | CORE API v3 `/search/works` | `fullText` field → download PDF → abstract |
| **web** | Brave Search API | URL filter by domain tier, then Firecrawl scraping |

The web agent's URL filter has two stages: (1) domain tier regex (`BLOCK`/`DEFAULT`/`GOOD`/`PRIORITY`), (2) optional LLM ranking of survivors using Brave snippets already in memory (disable with `web_llm_url_ranking: False` in config).

### `mara/merkle/` — Cryptographic integrity layer

Standalone sub-package with no intra-package imports. Can be imported and used independently of the agent system.

**`hasher.py`**
- `canonical_serialise(url, text, retrieved_at) -> bytes` — produces a byte-identical JSON encoding across Python versions, platforms, and locales (`sort_keys=True`, no whitespace, `ensure_ascii=True`). Any reader with the same three fields can recompute the hash independently.
- `hash_chunk(url, text, retrieved_at, algorithm) -> str` — returns the hex digest of `canonical_serialise(...)` using the given hashlib algorithm name (always `"sha256"` in production, from `ResearchConfig.hash_algorithm`).

**`tree.py`**
- `MerkleTree` — dataclass: `leaves`, `levels` (all tree levels; index 0 = leaves, last = `[root]`), `root`, `algorithm`.
- `build_merkle_tree(leaf_hashes, algorithm) -> MerkleTree` — balanced binary tree. Odd-length levels duplicate the last leaf. Returns an empty tree (root `""`) for zero leaves. Raises `ValueError` if any leaf hash is an empty string.

**`proof.py`**
- `generate_proof(tree, chunk_index) -> list[dict]` — returns the sibling hashes needed to reconstruct the root from a single leaf, as a list of `{"hash": ..., "direction": "left"|"right"}` steps.
- `verify_proof(leaf_hash, proof, root, algorithm) -> bool` — recomputes the root by walking the proof steps and returns `True` if it matches. Works without access to the full tree.

**`forest.py`**
- `ForestTree` — dataclass wrapping a `MerkleTree` whose leaves are the sub-tree roots from each `AgentFindings`.
- `build_forest_tree(findings, algorithm) -> ForestTree` — takes a sequence of `AgentFindings` (or any sequence of `(root, tree)` tuples), uses each sub-tree root as a leaf, and builds the meta-tree. Enables two-level proofs: chunk → sub-tree root, sub-tree root → forest root.

### Config (`mara/config.py`)

All tunable parameters live in `ResearchConfig` (pydantic-settings, env prefix `MARA_`). Never redeclare config values as loose function arguments or defaults.

### Build Order

Build in this sequence (later modules depend on earlier ones):

1. `merkle/` (hasher → tree → proof → forest)
2. `agents/types.py`, `agents/base.py`, `agents/registry.py`
3. Agent modules: `agents/arxiv/`, `agents/semantic_scholar/`, `agents/pubmed/`, `agents/core/`, `agents/web/`
4. `agent/nodes/corpus_assembler.py`
5. `agent/graph.py`

## Testing

### Coverage floor

Never let coverage drop below 98%. Enforced via `--cov-fail-under=98` in `addopts` — CI fails automatically. Use branch coverage (`--cov-branch`), not just line coverage. A function with `if x: return A` only needs one test for 100% line coverage — branch coverage requires both the true and false paths.

### Rules

**Mock at the boundary, not inside it.** Mock `httpx.AsyncClient.get`, not internal helpers that call it. Tests should break when real behavior changes, not when internal wiring is rearranged.

**Use `AsyncMock` for async code.** A regular `Mock` on an `async def` silently passes without awaiting. Always use `pytest-mock`'s `mocker.AsyncMock` for coroutines.

```python
# correct
mocker.patch("mara.agents.arxiv.downloader.httpx.AsyncClient.get",
             new_callable=AsyncMock, return_value=mock_response)

# wrong — test passes even if the coroutine is never awaited
mocker.patch("mara.agents.arxiv.downloader.httpx.AsyncClient.get",
             return_value=mock_response)
```

**Use `pytest.raises` with `match=`.**  Don't just assert an exception is raised — assert the message. This catches the right exception raised for the wrong reason.

```python
with pytest.raises(ValueError, match="merkle_root mismatch"):
    AgentFindings(agent_type="arxiv", ..., merkle_root="badhash", ...)
```

**Use `@pytest.mark.parametrize` to eliminate duplicate tests.** If two tests differ only in input values, parametrize them with `ids=` so failures are immediately readable.

**Use `yield` fixtures for anything with teardown.** Put setup before `yield`, teardown after. Scope: `session` for expensive resources, `function` (default) for everything else.

**One `conftest.py` per test directory.** Fixtures for `tests/merkle/` belong in `tests/merkle/conftest.py`, not the root `conftest.py`.

**Test behavior, not implementation.** A test that breaks when you rename a private method is a bad test. Test return values and raised exceptions — not which internal helpers are called, unless those calls are the explicit contract.

**Every test has exactly one reason to fail.** Split tests that assert unrelated things. Exception: multiple asserts on fields of a single returned object are fine.

**`# pragma: no cover` is a last resort.** Legitimate uses: `if TYPE_CHECKING:` blocks, `raise NotImplementedError` in abstract base classes. Any other use requires a comment explaining why.

**`asyncio_mode = "auto"`** means every `async def test_*` runs automatically without `@pytest.mark.asyncio`. Eliminates "test silently passed without awaiting" bugs.

## Required Environment Variables

```bash
BRAVE_API_KEY=...       # web agent discovery
HF_TOKEN=...            # LLM inference
FIRECRAWL_API_KEY=...   # web agent scraping
CORE_API_KEY=...        # CORE agent
S2_API_KEY=...          # recommended (Semantic Scholar)
NCBI_API_KEY=...        # optional (raises PubMed from 3 → 10 req/s)
```
