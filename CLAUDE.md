# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## TODO

- **Docker Containerization:** Create a Docker image with pandoc, Python 3.11+, and uv pre-installed to guarantee consistent LaTeX parsing availability across all deployments.

## Commands

```bash
# Install system dependency (pandoc — required for arxiv agent LaTeX parsing)
brew install pandoc          # macOS
apt install pandoc           # Linux (Debian/Ubuntu)

# Install Python dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/path/to/test_file.py

# Run tests with coverage (also enforced automatically via addopts)
uv run pytest --cov=mara --cov-branch --cov-report=term-missing --cov-fail-under=98

# Run the CLI
uv run mara "your research query"

# Run the CLI with JSON output
uv run mara --json "your research query"
```

## Architecture

MARA is a multi-agent research pipeline that produces cryptographically verifiable reports. Each agent retrieves raw source text, hashes it immediately, and assembles hashes into a Merkle sub-tree. Sub-tree roots become leaves of a meta-tree (`ForestTree`). Any chunk is verifiable with a two-level proof without re-running MARA.

**Core invariant:** `AgentFindings.__post_init__` recomputes the Merkle root from its chunks. Successful construction = verified. There is no external verification step.

### Data Flow

```text
query_planner → [route_to_agents] → run_agent (×N) → corpus_assembler → chunk_selector → report_synthesizer → certified_output
```

Fan-out uses LangGraph's `Send()` API. Fan-in uses `operator.add` on the state's `AgentFindings` list. The query planner assigns each sub-query to a specific agent (`SubQuery.agent`); routing is **directed** (1 Send per sub-query) when the LLM names a registered agent, or **broadcast** (1 Send per agent) as a fallback.

### `mara/agents/types.py` — Pipeline data types

- **`SubQuery`** — mutable dataclass: `query: str`, `domain: str = ""`, `agent: str = ""`. `query` is stripped; empty/whitespace-only raises `ValueError`. `agent` is set by the query planner for directed routing; empty string triggers broadcast fallback.
- **`RawChunk`** — mutable dataclass: `url`, `text`, `retrieved_at`, `source_type`, `sub_query`. Empty/whitespace `url` or `text` raises `ValueError`.
- **`VerifiedChunk`** — frozen dataclass: all `RawChunk` fields plus `hash` and `chunk_index` (agent-local; corpus assembler assigns global indices). `short_hash` returns first 8 hex chars. Produced exclusively by `SpecialistAgent.run()`.
- **`AgentFindings`** — frozen dataclass: `agent_type`, `query`, `chunks: tuple[VerifiedChunk, ...]`, `merkle_root`, `merkle_tree`. `__post_init__` recomputes the Merkle root and raises `ValueError("merkle_root mismatch")` if it doesn't match. `chunk_count` property returns `len(chunks)`.
- **`CertifiedReport`** — frozen dataclass: `original_query`, `report`, `forest_tree`, `chunks: tuple[VerifiedChunk, ...]`. The pipeline's final output. The `report` string includes inline citations (the LLM may produce `[ML:index:hash]`, `[N:hash]`, `[N]`, or `[N, M, ...]` format) followed by a `## References` section appended by `certified_output_node`.

### `mara/agents/registry.py` — Self-registration

`_REGISTRY: dict[str, AgentRegistration]` is the single source of truth for all agents.

- **`AgentConfig`** — dataclass: `api_key: str = ""`, `max_results: int = 20`, `rate_limit_rps: float = 0.0`, `max_concurrent: int = 0`, `retry_backoff_base: float = 0.0`, `max_retries: int = 0`, `max_sub_queries: int = 0`. `max_sub_queries`: 0 = unlimited; >0 caps how many sub-queries the planner may route to this agent. Holds per-agent tunable parameters. Each `AgentRegistration` has a default `config` field; runtime overrides come from `ResearchConfig.agent_config_overrides`.
- **`AgentRegistration`** — dataclass wrapping `cls`, `description`, `capabilities: list[str]`, `limitations: list[str]`, `example_queries: list[str]`, `config: AgentConfig`. All list fields default to `[]`. The `config` field holds the agent's default rate limits and per-agent settings.
- `@agent("name", description=..., capabilities=..., limitations=..., example_queries=..., config=...)` — class decorator that inserts an `AgentRegistration` into `_REGISTRY`. The `config` parameter (optional) sets the agent's default `AgentConfig`. Raises `ValueError` if the name is already taken.
- `get_agents(config)` — instantiates every registered class (`reg.cls`) with both `config` (the `ResearchConfig`) and the agent's `AgentConfig` (from `agent_config_overrides` or `reg.config`), returning the list.
- `get_registry_summary()` — returns a formatted multi-line string of the agent roster, injected into the query planner system prompt so the LLM can route sub-queries intelligently. Emits `Max sub-queries: N` per agent when `AgentConfig.max_sub_queries > 0`.

Adding a new agent requires writing the class, applying `@agent("name", ..., config=AgentConfig(...))`, and passing the agent to the constructor with both `ResearchConfig` and `AgentConfig` arguments — the pipeline wires them automatically via `get_agents()`.

### `mara/agents/cache.py` — Search result cache

- **`SearchCache`** — `@runtime_checkable` Protocol: `get(agent_type, query) -> list[RawChunk] | None` and `set(agent_type, query, result) -> None`. Both are `async`.
- **`NoOpCache`** — default. `get` always returns `None`; `set` is a no-op. Zero overhead when caching is not needed.
- **`InMemoryCache`** — simple dict-backed cache keyed by `(agent_type, query)`. Opt-in via `ResearchConfig(search_cache=InMemoryCache())`. Useful for repeated queries in the same process (e.g. interactive shells, tests).

`search_cache` is typed `Any` at runtime in `ResearchConfig` (same circular-import reason as `chunk_filter`); `TYPE_CHECKING` guard provides the `SearchCache` annotation.

### `mara/agents/base.py` — SpecialistAgent

Abstract base class all agents extend. Subclasses implement only `_search()`.

Pipeline: `_search() → _chunk() → _filter()` wired together in `_retrieve()`; `run()` calls `_retrieve()` and handles hashing.

**Rate limiting** — class-level infrastructure shared across all instances of the same agent type:

- `_rate_limit_interval: float = 0.0` — class variable. Set to a positive value for a fixed inter-call delay. Override `_get_rate_limit_interval()` instead for config-dependent values (e.g. `1.0 / agent_config.rate_limit_rps`).
- `_locks: dict[str, asyncio.Lock]` / `_last_called: dict[str, float]` — class-level dicts keyed by agent type string so distinct agent types don't interfere.
- `_get_lock(agent_type)` — classmethod; creates the lock on first access.
- `_reset_rate_limit_state()` — classmethod; clears both dicts. Call in test teardown fixtures.
- `_acquire_rate_limit_slot()` — blocks until it is safe to call `_search()` again. No-ops when interval ≤ 0. Logs `"rate limit: sleeping %.3fs"` at DEBUG only when a wait actually occurs.

**Retry** — `_fetch_with_retry(sub_query)`:

1. Check `config.search_cache.get(agent_type, query)` — cache hit skips rate limiting and the network call entirely.
2. Acquire rate-limit slot.
3. Loop `config.max_retries + 1` times:
   - Logs `"attempt %d/%d"` at DEBUG on each attempt.
   - On success: populate cache, return result.
   - `401`/`403` → logs WARNING `"auth error %d — aborting"`, re-raises immediately.
   - `404` → logs DEBUG `"404 — returning empty"`, returns `[]`.
   - `429`/`5xx` → logs WARNING `"HTTP %d on attempt %d — will retry: %s"` (includes the full exception string).
   - `ConnectError` / `TimeoutException` → logs WARNING with exception text.
   - Between retries: sleeps `retry_backoff_base ** attempt` seconds, logs DEBUG `"backing off %.1fs before attempt %d"`.
4. After all attempts exhausted: logs ERROR `"all %d attempt(s) failed"`, re-raises last exception.

All log calls include `extra={"agent": agent_type}` for structured JSON logging.

**Other methods:**

- **`_search(sub_query)`** — abstract. Fetch raw chunks. Must NOT hash, retry, or sleep.
- **`_chunk(raw)`** — concrete, overridable. Fixed-size character sliding window (`chunk_size`, step = `chunk_size − chunk_overlap`). Returns raw unchanged for degenerate config (`size ≤ 0` or `step ≤ 0`). Agents with pre-chunked content (ArXiv, S2, PubMed, CORE) override this to return `raw` unchanged.
- **`_filter(chunks, query)`** — concrete. Delegates entirely to `self.config.chunk_filter.filter(chunks, query)`.
- **`_retrieve(sub_query)`** — concrete pipeline. Logs per-stage counts at DEBUG.
- **`model()`** — returns `config.model_overrides.get(agent_type, config.default_model)`.
- **`run(sub_query)`** — public entrypoint. Calls `_retrieve()`, hashes each chunk, builds a `MerkleTree`, returns self-verified `AgentFindings`. Logs entry/exit at DEBUG.

Each agent module defines its own `source_type` string constants (e.g., `LATEX`, `ABSTRACT_ONLY`).

#### Agent Content Strategies

| Agent              | Discovery                                                                                                                                                                         | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **arxiv**          | `export.arxiv.org/api/query` (Atom XML, stdlib)                                                                                                                                   | `.tar.gz` → LaTeX → PDF in tarball → rendered PDF → abstract. **Note:** the `all:` prefix colon must not be percent-encoded — build the query string manually instead of passing `params=` to httpx. Sets `_rate_limit_interval = 3.0` (class variable) so the base class enforces 3 s between discovery calls. Additionally sleeps 3 s **inside `_search()`** between per-paper content fetches. `_chunk()` passes LATEX chunks through unchanged; applies the base sliding-window only to PDF/abstract chunks. source_types: `LATEX`, `PDF_RENDERED`, `ABSTRACT_ONLY`.                                                       |
| **s2**             | `api.semanticscholar.org/graph/v1/snippet/search`                                                                                                                                 | One `snippet` object per result item (not a list); paper identified by `corpusId`. Overrides `_get_rate_limit_interval()` returning `1.0 / agent_config.rate_limit_rps` (1 RPS by default). `_chunk()` passes snippets through unchanged.                                                                                                                                                                                                                                                                                                                                                                                      |
| **citation_graph** | Two-phase Semantic Scholar walk: (1) `GET {base}/paper/search?query={q}&limit=1` → extract `corpusId`; (2) `GET {base}/paper/CorpusId:{id}/citations?limit={max}` → citing papers | Retrieves papers citing a seed paper. Overrides `_get_rate_limit_interval()` returning `1.0 / agent_config.rate_limit_rps` (1 RPS by default). Early return if Phase 1 finds no seed paper. Sends `x-api-key` header in both phases. Text format: `{year}: {title}\n\n{abstract}` (degrades gracefully; skips items with no title and no abstract). `_chunk()` passes through unchanged. source_type: `CITATION`. URL: `https://www.semanticscholar.org/paper/{corpusId}`.                                                                                                                                                     |
| **pubmed**         | NCBI `esearch.fcgi` (PMID list) + `esummary.fcgi` (metadata + PMC ID)                                                                                                             | `efetch.fcgi?db=pmc` → parses `<sec>` elements (full text); falls back to `efetch.fcgi?db=pubmed` → `<AbstractText>` (abstract). Overrides `_get_rate_limit_interval()` returning `1.0 / agent_config.rate_limit_rps` (3 RPS by default, 10 RPS with NCBI API key). Also sleeps `delay` seconds **inside `_search()`** between each eUtils HTTP call (esearch, esummary, each efetch) for intra-request pacing. `_ncbi_params()` helper omits `api_key` from params when `agent_config.api_key` is blank (NCBI returns 400 otherwise). `_chunk()` passes sections through unchanged. source_types: `PMC_XML`, `ABSTRACT_ONLY`. |
| **core**           | CORE API v3 `/search/works/` (trailing slash — API redirects; use `follow_redirects=True`)                                                                                        | `fullText` field → `_fetch_pdf_text()` module-level helper downloads PDF → falls back to abstract. No rate-limit override (base default 0.0). `_chunk()` passes through unchanged. source_types: `FULLTEXT`, `PDF_DOWNLOADED`, `ABSTRACT_ONLY`.                                                                                                                                                                                                                                                                                                                                                                                |
| **pwc**            | `api.huggingface.co/papers?q={query}&limit={N}`                                                                                                                                   | Paper abstracts from HuggingFace Papers (formerly Papers With Code). Returns title + summary per paper. No rate-limit override (base default 0.0). `_chunk()` passes abstracts through unchanged. No API key required. source_type: `PAPER_ABSTRACT`.                                                                                                                                                                                                                                                                                                                                                                          |
| **biorxiv**        | `api.biorxiv.org/details/{server}/{start}/{end}/0` (90-day window, both servers)                                                                                                  | Fetches last 90 days from both bioRxiv and medRxiv servers, then filters client-side by keyword match against title and abstract. No rate-limit override (uses 30-second timeout for bulk fetch). `_chunk()` passes preprints through unchanged. No API key required. source_types: `BIORXIV`, `MEDRXIV`.                                                                                                                                                                                                                                                                                                                      |
| **nber**           | `https://www.nber.org/api/v1/working_page_listing/contentType/working_paper/_/_/search`                                                                                           | Uses `curl_cffi.requests.AsyncSession` (not httpx). Returns working paper metadata; text format: `"{date}: {title}\n\nAuthors: ...\n\n{abstract}"`. Overrides `_get_rate_limit_interval()` at 1 RPS. `_chunk()` passes papers through unchanged. No API key required. source_type: `working_paper`.                                                                                                                                                                                                                                                                                                                            |
| **web**            | Brave Search API                                                                                                                                                                  | URL filter by domain tier, then Firecrawl scraping                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

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

### `mara/agent/` — LangGraph pipeline

**`mara/agent/state.py`**

- **`GraphState`** — `TypedDict(total=False)`: `original_query`, `sub_queries`, `findings` (reduced with `operator.add`), `forest_tree`, `flattened_chunks`, `selected_chunks`, `report`, `certified_report`. All keys optional so nodes return partial updates.
- `selected_chunks`: `list[VerifiedChunk]` — deduplicated, BM25-ranked subset of `flattened_chunks`, capped at `chunk_selector_cap`. Written by `chunk_selector_node`; preferred by `report_synthesizer` over `flattened_chunks`.
- **`AgentRunState`** — `TypedDict`: `sub_query`, `agent_type`. Payload for each `Send()` fan-out invocation.

**`mara/agent/graph.py`**

- **`build_graph()`** — compiles the `StateGraph`. Topology: `START → query_planner → route_to_agents → run_agent (×N×M) → corpus_assembler → chunk_selector → report_synthesizer → certified_output → END`.
- **`run_research(query, config)`** — async entry point. Calls `build_graph()`, invokes with `{"original_query": query}`, passes `config` via `{"configurable": {"research_config": config}}`. Returns `CertifiedReport`. **Agent modules must be imported before this is called** (the CLI does this in `run()`).

**`mara/agent/edges/routing.py`**

- **`route_to_agents(state, config)`** — conditional edge from `query_planner`. Directed routing: if `sub_query.agent` names a registered agent, emits one `Send` to that agent only. Broadcast fallback: if `sub_query.agent` is empty or unrecognized, emits one `Send` per registered agent. Returns `"corpus_assembler"` (bypass string edge) when sub-queries or registry is empty.

**`mara/agent/nodes/`**

- **`query_planner_node`** — builds a dynamic system prompt via `get_registry_summary()` (includes routing constraints: respect `Max sub-queries` limits, require distinct aspects, prefer concrete searchable phrases), calls LLM, parses JSON array of `{query, domain, agent}` objects into `SubQuery` list. Falls back to single `SubQuery(original_query)` on parse failure.
- **`run_agent_node`** — looks up `AgentRegistration` from `_REGISTRY` by `agent_type`, instantiates `reg.cls` with `research_config`, calls `agent.run(sub_query)`. Returns `{"findings": [AgentFindings]}`. On exception, logs a warning and returns `{"findings": []}` so one agent failure does not crash the pipeline.
- **`corpus_assembler_node`** — groups chunks by `agent_type`, sorts by `(sub_query, chunk_index)`, skips agents with 0 chunks (empty root is not a valid forest leaf), builds one `MerkleTree` root per agent, calls `build_forest_tree`, assigns globally unique `chunk_index` values via `dataclasses.replace`.
- **`chunk_selector_node`** — deduplicates `flattened_chunks` by hash, BM25Plus-ranks against `original_query` (via `mara/agent/scoring.py`), caps at `research_config.chunk_selector_cap` (default 50), writes `selected_chunks`.
- **`report_synthesizer_node`** — reads `selected_chunks` (preferred) or `flattened_chunks` (fallback), formats chunks as `[index:short_hash] text`, calls LLM to synthesise a structured report (title, executive summary, thematic sections, conclusion; 800–1500 words) with inline `[ML:index:hash]` citations. Prompt instructs model to flag thin evidence and acknowledge conflicting sources explicitly.
- **`certified_output_node`** — parses citation indices from the report (handling `[ML:N:hash]`, `[N:hash]`, `[N]`, and `[N, M, ...]` formats — the LLM may omit the `ML:` prefix), appends a `## References` section mapping each cited index to its source URL, then packages `original_query`, `report`, `forest_tree`, and `flattened_chunks` into a `CertifiedReport`. The references section is omitted when no valid citations are found.

### Config (`mara/config.py`)

All tunable parameters live in `ResearchConfig` (pydantic-settings, no env prefix — env vars use plain names: `BRAVE_API_KEY`, `HF_TOKEN`, etc.). Never redeclare config values as loose function arguments or defaults.

**Per-agent configuration** — `agent_config_overrides: dict[str, AgentConfig]` holds runtime overrides keyed by agent name (e.g., `{"s2": AgentConfig(api_key="...", rate_limit_rps=0.5)}`). A model validator `_wire_api_keys()` automatically pushes flat API keys (`s2_api_key`, `core_api_key`, `ncbi_api_key`) into the overrides for the respective agents. `s2_api_key` is wired to both `s2` and `citation_graph` agents. Defaults come from each `AgentRegistration.config`; `get_agents()` merges registry defaults with runtime overrides at instantiation.

`chunk_filter` is typed `Any` at runtime to avoid a circular import (`config → agents.filtering → agents.__init__ → agents.base → config`); a `TYPE_CHECKING` guard provides the `ChunkFilter` annotation for static analysis. The default `CapFilter()` is set lazily via `model_validator(mode="after")`.

`search_cache` follows the same pattern: typed `Any` at runtime, `SearchCache` under `TYPE_CHECKING`, default `NoOpCache()` set by the same `model_validator`.

`chunk_selector_cap: int = 50` — maximum number of chunks passed from `chunk_selector` to `report_synthesizer`. Controls context window usage.

### Logging (`mara/logging.py`)

`configure_logging(log_level)` sets up a single `mara` root logger with a JSON formatter on stderr. All submodule loggers (e.g. `mara.agents.base`, `mara.agents.registry`) are children and inherit this handler. Call once at startup.

### Build Order

1. `merkle/` (hasher → tree → proof → forest)
2. `agents/types.py`, `agents/filtering.py`, `agents/base.py`, `agents/registry.py`
3. Agent modules: `agents/arxiv/`, `agents/semantic_scholar/`, `agents/citation_graph/`, `agents/pubmed/`, `agents/core/`, `agents/pwc/`, `agents/biorxiv/`, `agents/web/`
4. `agent/nodes/corpus_assembler.py`, `agent/scoring.py`, `agent/nodes/chunk_selector.py`
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
