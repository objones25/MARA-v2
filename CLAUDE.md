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

- **`SubQuery`** — mutable dataclass: `query: str`, `domain: str = ""`. `query` is stripped; empty/whitespace-only raises `ValueError`.
- **`RawChunk`** — mutable dataclass: `url`, `text`, `retrieved_at`, `source_type`, `sub_query`. Empty/whitespace `url` or `text` raises `ValueError`.
- **`VerifiedChunk`** — frozen dataclass: all `RawChunk` fields plus `hash` and `chunk_index` (agent-local; corpus assembler assigns global indices). `short_hash` returns first 8 hex chars. Produced exclusively by `SpecialistAgent.run()`.
- **`AgentFindings`** — frozen dataclass: `agent_type`, `query`, `chunks: tuple[VerifiedChunk, ...]`, `merkle_root`, `merkle_tree`. `__post_init__` recomputes the Merkle root and raises `ValueError("merkle_root mismatch")` if it doesn't match. `chunk_count` property returns `len(chunks)`.

### `mara/agents/registry.py` — Self-registration

`_REGISTRY: dict[str, type[SpecialistAgent]]` is the single source of truth for all agents.

- `@agent("name")` — class decorator that inserts the class into `_REGISTRY`. Raises `ValueError` if the name is already taken.
- `get_agents(config)` — instantiates every registered class with `config` and returns the list.

Adding a new agent requires only writing the class and applying `@agent("name")` — nothing else in the pipeline changes.

### `mara/agents/base.py` — SpecialistAgent

Abstract base class all agents extend. Subclasses implement only `_search()`.

Pipeline: `_search() → _chunk() → _filter()` wired together in `_retrieve()`; `run()` calls `_retrieve()` and handles hashing.

- **`_search(sub_query)`** — abstract. Fetch raw chunks. Must NOT hash.
- **`_chunk(raw)`** — concrete, overridable. Fixed-size character sliding window (`chunk_size`, step = `chunk_size − chunk_overlap`). Returns raw unchanged for degenerate config (`size ≤ 0` or `step ≤ 0`). Agents with pre-chunked content (ArXiv, S2, PubMed) override this.
- **`_filter(chunks, query)`** — concrete. Delegates entirely to `self.config.chunk_filter.filter(chunks, query)`.
- **`_retrieve(sub_query)`** — concrete pipeline. Logs per-stage counts at DEBUG.
- **`model()`** — returns `config.model_overrides.get(agent_type, config.default_model)`.
- **`run(sub_query)`** — public entrypoint. Calls `_retrieve()`, hashes each chunk, builds a `MerkleTree`, returns self-verified `AgentFindings`. Logs entry/exit at DEBUG.

Each agent module defines its own `source_type` string constants (e.g., `LATEX`, `ABSTRACT_ONLY`).

#### Agent Content Strategies

| Agent      | Discovery                                                             | Content                                                                                                                                                                                                                                                                                                                   |
| ---------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **arxiv**  | `export.arxiv.org/api/query` (Atom XML, stdlib)                       | `.tar.gz` → LaTeX → PDF in tarball → rendered PDF → abstract. **Note:** the `all:` prefix colon must not be percent-encoded — build the query string manually instead of passing `params=` to httpx.                                                                                                                      |
| **s2**     | `api.semanticscholar.org/graph/v1/snippet/search`                     | One `snippet` object per result item (not a list); paper identified by `corpusId`. Rate-limited to ≤1 RPS via shared `asyncio.Lock` singleton. `_chunk()` passes snippets through unchanged.                                                                                                                              |
| **pubmed** | NCBI `esearch.fcgi` (PMID list) + `esummary.fcgi` (metadata + PMC ID) | `efetch.fcgi?db=pmc` → parses `<sec>` elements (full text); falls back to `efetch.fcgi?db=pubmed` → `<AbstractText>` (abstract). Rate-limited via shared `asyncio.Lock` singleton. `api_key` omitted from params when `ncbi_api_key` is blank (NCBI returns 400 otherwise). `_chunk()` passes sections through unchanged. |
| **core**   | CORE API v3 `/search/works`                                           | `fullText` field → download PDF → abstract                                                                                                                                                                                                                                                                                |
| **web**    | Brave Search API                                                      | URL filter by domain tier, then Firecrawl scraping                                                                                                                                                                                                                                                                        |

The web agent's URL filter has two stages: (1) domain tier regex (`BLOCK`/`DEFAULT`/`GOOD`/`PRIORITY`), (2) optional LLM ranking of survivors (disable with `web_llm_url_ranking: False`).

### `mara/agents/filtering.py` — Chunk filter strategies

- **`ChunkFilter`** — `@runtime_checkable` Protocol with a single method `filter(chunks, query) -> list[RawChunk]`. `@runtime_checkable` lets pydantic validate instances with `isinstance`.
- **`CapFilter`** — dataclass. Applies a per-URL cap (`max_chunks_per_url=3`) then a global cap (`max_chunks_per_agent=50`). Default filter in `ResearchConfig`.
- **`EmbeddingFilter`** — stub. Raises `NotImplementedError`. Fields: `model`, `similarity_threshold`, `max_chunks_per_agent`.

### `mara/merkle/` — Cryptographic integrity layer

Standalone sub-package with no intra-package imports.

- **`hasher.py`**: `canonical_serialise(url, text, retrieved_at) -> bytes` (deterministic JSON); `hash_chunk(..., algorithm) -> str` hex digest.
- **`tree.py`**: `MerkleTree` dataclass; `build_merkle_tree(leaf_hashes, algorithm)` — balanced binary tree, odd levels duplicate last leaf, empty input → root `""`.
- **`proof.py`**: `generate_proof(tree, chunk_index)` → sibling list; `verify_proof(leaf_hash, proof, root, algorithm) -> bool`.
- **`forest.py`**: `ForestTree` wrapping a `MerkleTree` of sub-tree roots; `build_forest_tree(findings, algorithm)` enables two-level proofs.

### Config (`mara/config.py`)

All tunable parameters live in `ResearchConfig` (pydantic-settings, no env prefix — env vars use plain names: `BRAVE_API_KEY`, `HF_TOKEN`, etc.). Never redeclare config values as loose function arguments or defaults.

`chunk_filter` is typed `Any` at runtime to avoid a circular import (`config → agents.filtering → agents.__init__ → agents.base → config`); a `TYPE_CHECKING` guard provides the `ChunkFilter` annotation for static analysis. The default `CapFilter()` is set lazily via `model_validator(mode="after")`.

### Logging (`mara/logging.py`)

`configure_logging(log_level)` sets up a single `mara` root logger with a JSON formatter on stderr. All submodule loggers (e.g. `mara.agents.base`, `mara.agents.registry`) are children and inherit this handler. Call once at startup.

### Build Order

1. `merkle/` (hasher → tree → proof → forest)
2. `agents/types.py`, `agents/filtering.py`, `agents/base.py`, `agents/registry.py`
3. Agent modules: `agents/arxiv/`, `agents/semantic_scholar/`, `agents/pubmed/`, `agents/core/`, `agents/web/`
4. `agent/nodes/corpus_assembler.py`
5. `agent/graph.py`

## Testing

### Coverage floor

Never let coverage drop below 98%. Enforced via `--cov-fail-under=98` in `addopts`. Use branch coverage (`--cov-branch`) — line coverage alone misses untaken branches.

### Rules

**Mock at the boundary.** Mock `httpx.AsyncClient.get`, not internal helpers that call it.

**Use `AsyncMock` for async code.** A regular `Mock` on an `async def` silently passes without awaiting.

**Use `pytest.raises` with `match=`.** Assert the message, not just the exception type.

**Use `@pytest.mark.parametrize` to eliminate duplicate tests.** Add `ids=` so failures are readable.

**Use `yield` fixtures for teardown.** Scope: `session` for expensive resources, `function` for everything else.

**One `conftest.py` per test directory.**

**Test behavior, not implementation.** Test return values and raised exceptions — not which internal helpers are called.

**Every test has exactly one reason to fail.**

**`# pragma: no cover` is a last resort.** Legitimate uses: `if TYPE_CHECKING:` blocks, `raise NotImplementedError` in abstract base classes.

**`asyncio_mode = "auto"`** — every `async def test_*` runs without `@pytest.mark.asyncio`.

## Required Environment Variables

```bash
BRAVE_API_KEY=...       # web agent discovery
HF_TOKEN=...            # LLM inference
FIRECRAWL_API_KEY=...   # web agent scraping
CORE_API_KEY=...        # CORE agent
S2_API_KEY=...          # Semantic Scholar
NCBI_API_KEY=...        # PubMed (raises rate limit 3 → 10 req/s)
```
