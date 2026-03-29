# MARA Agents Package Documentation

**Last Updated:** 2026-03-29

## Table of Contents

1. [Package Overview](#package-overview)
2. [Data Types](#data-types)
3. [SpecialistAgent Base Class](#specialistagent-base-class)
4. [Agent Registry](#agent-registry)
5. [Cache Strategies](#cache-strategies)
6. [Chunk Filtering](#chunk-filtering)
7. [Specialist Agents](#specialist-agents)
   - [ArXiv Agent](#arxiv-agent)
   - [Semantic Scholar Agent](#semantic-scholar-agent)
   - [PubMed Agent](#pubmed-agent)
   - [CORE Agent](#core-agent)
   - [Web Agent](#web-agent)
8. [Adding a New Agent](#adding-a-new-agent)

---

## Package Overview

The `mara/agents/` package contains all specialist agents that retrieve research information from various sources. Each agent:

- Extends the `SpecialistAgent` abstract base class
- Implements the `_search()` method to fetch raw content
- Is self-registered via the `@agent()` decorator
- Participates in the LangGraph pipeline for multi-agent research orchestration

**Module Map:**

```text
mara/agents/
├── types.py           # Core data types (SubQuery, RawChunk, VerifiedChunk, AgentFindings, CertifiedReport)
├── base.py            # SpecialistAgent abstract base class
├── registry.py        # Agent registration and lookup
├── cache.py           # Search result caching strategies
├── filtering.py       # Chunk filtering strategies
├── arxiv/             # ArXiv agent (LaTeX, PDF, abstract fallback)
│   ├── agent.py
│   ├── fetcher.py
│   └── latex_parser.py
├── semantic_scholar/  # Semantic Scholar agent (snippets)
│   └── agent.py
├── pubmed/            # PubMed agent (PMC full-text, abstract fallback)
│   ├── agent.py
│   ├── search.py
│   └── fulltext.py
├── core/              # CORE agent (fulltext, PDF, abstract fallback)
│   ├── agent.py
│   └── fetcher.py
└── web/               # Web agent (Brave Search + Firecrawl + filtering)
    ├── agent.py
    ├── scraper.py
    └── url_filter.py
```

---

## Data Types

All data types are defined in `mara/agents/types.py`. These immutable/frozen structures form the contract between pipeline stages.

### SubQuery

A decomposed research question, created by the query planner.

```python
@dataclass
class SubQuery:
    query: str          # Stripped of leading/trailing whitespace; empty raises ValueError
    domain: str = ""    # Optional hint (e.g. "empirical", "clinical") for the LLM
    agent: str = ""     # Set by query planner; empty string triggers broadcast fallback
```

**Validation:**

- `query` is stripped on construction; empty or whitespace-only raises `ValueError("SubQuery.query must not be empty or whitespace")`

**Usage:**

- Passed to every agent's `_search()` method
- `agent` field enables directed routing (1 Send to named agent) vs. broadcast (1 Send to all agents)

---

### RawChunk

Unverified source content returned by an agent's `_retrieve()` method (before hashing).

```python
@dataclass
class RawChunk:
    url: str            # Source URL; empty/whitespace raises ValueError
    text: str           # Content; empty/whitespace raises ValueError
    retrieved_at: str   # ISO-8601 UTC timestamp
    source_type: str    # Agent-specific label (e.g. "latex", "snippet", "pmc_xml")
    sub_query: str      # The sub-query that produced this chunk
```

**Validation:**

- `url` and `text` must not be empty or whitespace-only; validation happens in `__post_init__`
- Prevents garbage data from reaching the hasher

**Source Type Constants by Agent:**

| Agent      | Source Types                                                 |
| ---------- | ------------------------------------------------------------ |
| **arxiv**  | `LATEX`, `PDF_FROM_TARBALL`, `PDF_RENDERED`, `ABSTRACT_ONLY` |
| **s2**     | `SNIPPET`                                                    |
| **pubmed** | `PMC_XML`, `ABSTRACT_ONLY`                                   |
| **core**   | `FULLTEXT`, `PDF_DOWNLOADED`, `ABSTRACT_ONLY`                |
| **web**    | `WEB`                                                        |

---

### VerifiedChunk

Source content with cryptographic hash. **Produced exclusively by `SpecialistAgent.run()`.**

```python
@dataclass(frozen=True)
class VerifiedChunk:
    hash: str           # SHA-256 hex digest of canonical_serialise(url, text, retrieved_at)
    url: str
    text: str
    retrieved_at: str   # ISO-8601 UTC timestamp
    source_type: str
    sub_query: str
    chunk_index: int    # Position within agent's chunk list for this sub-query (NOT global)

    @property
    def short_hash(self) -> str:
        """First 8 hex characters, used in citations and logging."""
        return self.hash[:8]
```

**Key Points:**

- Immutable (frozen dataclass)
- `chunk_index` is agent-local; the corpus assembler assigns global indices during flattening
- Never constructed manually — only by `SpecialistAgent.run()`

---

### AgentFindings

Verified output from one specialist agent for one sub-query. Contains a self-verified Merkle tree.

```python
@dataclass(frozen=True)
class AgentFindings:
    agent_type: str                     # Registry name of the agent
    query: str                          # The sub-query string
    chunks: tuple[VerifiedChunk, ...]   # All verified chunks for this query
    merkle_root: str                    # Root hash of the Merkle tree
    merkle_tree: MerkleTree             # Full tree structure (not used in comparisons)

    def __post_init__(self) -> None:
        # Recomputes merkle_root from chunks; raises ValueError if mismatch
        ...

    @property
    def chunk_count(self) -> int:
        """Number of verified chunks."""
        return len(self.chunks)
```

**Core Invariant:**

- `__post_init__` recomputes the Merkle root and raises `ValueError("merkle_root mismatch")` if the provided root doesn't match
- **Successful construction of `AgentFindings` = verification of its chunk list**
- No external verification step needed

---

### CertifiedReport

The final output of the pipeline—a report bundled with full cryptographic provenance.

```python
@dataclass(frozen=True)
class CertifiedReport:
    original_query: str                 # The original user query
    report: str                         # Synthesized report with inline citations
    forest_tree: ForestTree             # Meta-tree of all agent sub-trees
    chunks: tuple[VerifiedChunk, ...]   # Flattened, globally-indexed chunks
```

**Citation Format:**

- Report includes inline citations like `[ML:index:hash]` where `index` is the chunk's global position and `hash` is its `short_hash`
- The LLM may also produce `[N:hash]`, `[N]`, or `[N, M, ...]` formats
- `certified_output_node` parses these and appends a `## References` section mapping indices to URLs

---

## SpecialistAgent Base Class

All specialist agents inherit from `SpecialistAgent` (in `mara/agents/base.py`). Subclasses need only implement `_search()`.

### Class Hierarchy

```text
SpecialistAgent (abstract base)
├── ArxivAgent
├── SemanticScholarAgent
├── PubMedAgent
├── COREAgent
└── WebAgent
```

### Constructor

```python
def __init__(self, config: ResearchConfig, agent_config: AgentConfig) -> None:
    self.config = config
    self.agent_config = agent_config
    self._cached_agent_type: str = ...  # Looked up from registry
```

**Key Attributes:**

- `config` — The `ResearchConfig` instance (global pipeline settings)
- `agent_config` — The `AgentConfig` instance for this agent (rate limits, API key, max results)
- `_cached_agent_type` — The agent's registry name (looked up from `_REGISTRY`)

### String Representations

```python
def __repr__(self) -> str:
    return f"{type(self).__name__}(model={self.model()!r})"

def __str__(self) -> str:
    return self._cached_agent_type
```

---

### Pipeline Stages

The agent retrieves data through a standard pipeline:

```text
_search() → _chunk() → _filter()  [orchestrated by _retrieve()]
_retrieve()                        [called by run()]
run()                             [public entrypoint]
```

#### \_search(sub_query: SubQuery) → list[RawChunk]

**Abstract method. Subclasses implement this.**

- Pure HTTP fetch — no locks, no retries, no sleeps
- Returns raw chunks with no processing
- All error handling, rate limiting, and retry logic is in the base class
- May raise `httpx.HTTPError` which is caught by `_fetch_with_retry()`

Example:

```python
async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={"q": sub_query.query})
        resp.raise_for_status()
        # Parse and return RawChunk list
        return chunks
```

#### \_chunk(raw: list[RawChunk]) → list[RawChunk]

**Concrete, overridable. Default: fixed-size character sliding window.**

```python
def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
    """Split raw chunks using config.chunk_size and config.chunk_overlap.

    Returns raw unchanged when chunk_size ≤ 0 or overlap ≥ size (degenerate config).
    Agents with pre-chunked content should override and return raw as-is or apply
    domain-aware splitting.
    """
```

**Default Behavior:**

- Window size: `config.chunk_size`
- Step: `config.chunk_size - config.chunk_overlap`
- Chunks smaller than window are returned unchanged

**Override Pattern** (e.g., for LaTeX sections):

```python
def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
    latex = [c for c in raw if c.source_type == LATEX]
    other = [c for c in raw if c.source_type != LATEX]
    return latex + (super()._chunk(other) if other else [])
```

#### \_filter(chunks: list[RawChunk], query: str) → list[RawChunk]

**Concrete. Delegates to `config.chunk_filter.filter()`.**

```python
def _filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
    """Filter chunks using the configured strategy."""
    chunk_filter: ChunkFilter = self.config.chunk_filter
    return chunk_filter.filter(chunks, query)
```

#### \_retrieve(sub_query: SubQuery) → list[RawChunk]

**Concrete. Orchestrates the pipeline stages.**

```python
async def _retrieve(self, sub_query: SubQuery) -> list[RawChunk]:
    """Pipeline: _fetch_with_retry() → _chunk() → _filter()."""
    raw = await self._fetch_with_retry(sub_query)
    chunks = self._chunk(raw)
    filtered = self._filter(chunks, sub_query.query)
    # Logs counts at DEBUG for visibility
    return filtered
```

#### run(sub_query: SubQuery) → AgentFindings

**Public entrypoint. Hashes chunks and assembles `AgentFindings`.**

```python
async def run(self, sub_query: SubQuery) -> AgentFindings:
    """Retrieve, hash, and package source chunks into verified AgentFindings.

    Calls _retrieve(), hashes every chunk, builds a Merkle tree, and returns
    a self-verified AgentFindings.
    """
```

**Process:**

1. Calls `_retrieve()`
2. Hashes each chunk with `hash_chunk(url, text, retrieved_at, algorithm)`
3. Wraps each hash in a `VerifiedChunk` with a local `chunk_index`
4. Builds a `MerkleTree` from the hashes
5. Constructs and returns `AgentFindings` (which verifies the tree)
6. Logs entry/exit at DEBUG level

---

### Rate Limiting

**Class-level state** shared across all instances of the same agent type:

```python
_rate_limit_interval: float = 0.0     # Class variable; 0.0 = no rate limit
_locks: dict[str, asyncio.Lock] = {}  # Per-agent-type lock
_last_called: dict[str, float] = {}    # Per-agent-type last call timestamp
```

#### \_get_rate_limit_interval() → float

**Concrete, overridable. Checks `agent_config.rate_limit_rps` first, falls back to `_rate_limit_interval`.**

```python
def _get_rate_limit_interval(self) -> float:
    """Return minimum seconds between consecutive _search() calls.

    Reads agent_config.rate_limit_rps when set (> 0), otherwise falls back to
    the class-level _rate_limit_interval variable.

    Example: S2 agent sets agent_config.rate_limit_rps=1.0 to enforce 1 RPS.
    """
    if self.agent_config.rate_limit_rps > 0.0:
        return 1.0 / self.agent_config.rate_limit_rps
    return self._rate_limit_interval
```

#### \_acquire_rate_limit_slot() → Coroutine[None]

**Async. Blocks until it is safe to call `_search()` again.**

```python
async def _acquire_rate_limit_slot(self) -> None:
    """Enforce _get_rate_limit_interval() seconds between calls.

    No-op when interval ≤ 0.
    Logs "rate limit: sleeping %.3fs" at DEBUG only when a wait occurs.
    Handles ZeroDivisionError from config-dependent intervals gracefully.
    """
```

#### \_get_lock(agent_type: str) → asyncio.Lock

**Classmethod. Returns (creating if absent) the per-agent-type lock.**

#### \_reset_rate_limit_state() → None

**Classmethod. Clears all rate-limit state.**

**Usage in tests:**

```python
@pytest.fixture(scope="function")
def reset_rate_limits():
    yield
    SpecialistAgent._reset_rate_limit_state()
```

---

### Retry Mechanism

#### \_fetch_with_retry(sub_query: SubQuery) → list[RawChunk]

**Orchestrates cache checking, rate limiting, and exponential backoff.**

```python
async def _fetch_with_retry(self, sub_query: SubQuery) -> list[RawChunk]:
    """Call _search() with caching, rate limiting, and exponential back-off.

    1. Check cache — hit skips rate limiting and network call
    2. Acquire rate-limit slot
    3. Loop max_retries + 1 times:
       - 401/403 → log WARNING, re-raise (auth failure)
       - 404 → log DEBUG, return [] (not found)
       - 429/5xx → log WARNING, retry with exponential backoff
       - ConnectError/TimeoutException → log WARNING, retry
    4. Populate cache on success
    5. Re-raise final exception if all retries exhausted
    """
```

**Error Classification:**

| Status/Exception   | Behavior                                                      |
| ------------------ | ------------------------------------------------------------- |
| `401`, `403`       | Permanent auth failure; re-raise immediately (WARNING logged) |
| `404`              | Not found; return `[]` (DEBUG logged)                         |
| `429`, `5xx`       | Transient; retry with backoff (WARNING logged)                |
| `ConnectError`     | Transient; retry with backoff (WARNING logged)                |
| `TimeoutException` | Transient; retry with backoff (WARNING logged)                |

**Backoff Formula:**

- Sleep duration: `retry_backoff_base ** attempt`
- Default `retry_backoff_base = 2.0` (config-driven)
- Example with default: attempt 0→2s, 1→4s, 2→8s, etc.

**Log Output:**

- All log calls include `extra={"agent": agent_type}` for structured JSON logging
- Detailed messages at DEBUG level for debugging
- Warnings and errors for important events

---

### Model Selection

#### model() → str

**Returns the LLM model for this agent.**

```python
def model(self) -> str:
    """Return the model to use for this agent.

    Falls back to config.default_model when no per-agent override is set.
    """
    return self.config.model_overrides.get(self._agent_type(), self.config.default_model)
```

**Usage:**

- Agents can override the default LLM per agent type
- Configured via `ResearchConfig.model_overrides: dict[str, str]`
- Example: `model_overrides={"arxiv": "claude-3-5-sonnet-20241022"}`

---

### Agent Identity

#### \_agent_type() → str

**Returns the registry name for this agent class.**

```python
def _agent_type(self) -> str:
    """Return the registry name for this agent class."""
    return self._cached_agent_type
```

---

## Agent Registry

The registry (`mara/agents/registry.py`) is the single source of truth for all agents. It enables self-registration and runtime discovery.

### AgentConfig

Per-agent tunable parameters, stored as defaults on `AgentRegistration` and merged with runtime overrides at agent instantiation.

```python
@dataclass
class AgentConfig:
    api_key: str = ""              # API key for this agent's service
    max_results: int = 20          # Max search results per query
    rate_limit_rps: float = 0.0    # Rate limit in requests/second (0 = no limit)
```

**Usage:**

- Each agent registers with a default `AgentConfig` (e.g., PubMed with `rate_limit_rps=3.0`)
- Runtime overrides come from `ResearchConfig.agent_config_overrides: dict[str, AgentConfig]`
- When instantiating, `get_agents()` merges the registry default with any runtime override
- Agents access their config via `self.agent_config` (e.g., `self.agent_config.api_key`)

### AgentRegistration

Metadata for a registered agent, including its default configuration.

```python
@dataclass
class AgentRegistration:
    cls: type[SpecialistAgent]      # The agent class
    description: str = ""            # One-sentence summary
    capabilities: list[str] = ...    # What it does well
    limitations: list[str] = ...     # Known weaknesses
    example_queries: list[str] = ... # Representative sub-queries
    config: AgentConfig = ...        # Default AgentConfig for this agent
```

**All list fields default to `[]`; `config` defaults to a new `AgentConfig()`.**

### @agent() Decorator

Registers a `SpecialistAgent` subclass by name with optional per-agent configuration.

```python
@agent(
    name="pubmed",
    description="Retrieves biomedical and clinical research papers from PubMed/PMC.",
    capabilities=[
        "Full-text PMC articles when available; abstract fallback",
        "Authoritative source for clinical trials and medical research",
    ],
    limitations=[
        "Limited to biomedical/life sciences",
        "Full text only available for open-access PMC articles",
    ],
    example_queries=[
        "CRISPR-Cas9 off-target effects in vivo",
        "efficacy of mRNA vaccines against SARS-CoV-2",
    ],
    config=AgentConfig(rate_limit_rps=3.0),  # 3 requests/second by default
)
class PubMedAgent(SpecialistAgent):
    ...
```

**Behavior:**

- Inserts an `AgentRegistration` into `_REGISTRY` with the provided defaults
- Raises `ValueError` if the name is already taken
- Logs at DEBUG: `"registered agent 'name' → ClassName"`

**Arguments:**

- `name` (str) — Unique agent identifier used throughout the pipeline
- `description` (str, optional) — One-sentence summary
- `capabilities` (list[str], optional) — What this agent does well
- `limitations` (list[str], optional) — Known weaknesses
- `example_queries` (list[str], optional) — Representative sub-queries
- `config` (AgentConfig, optional) — Default `AgentConfig` (rate limits, max results, API key defaults)

---

### get_agents(config: ResearchConfig) → list[SpecialistAgent]

**Instantiate every registered agent with both `ResearchConfig` and merged `AgentConfig`.**

```python
def get_agents(config: ResearchConfig) -> list[SpecialistAgent]:
    """Instantiate every registered agent with config and agent-specific overrides.

    For each registered agent:
    1. Look up runtime overrides from config.agent_config_overrides[agent_name]
    2. Fall back to reg.config (the registry default)
    3. Instantiate with both ResearchConfig and the merged AgentConfig
    """
    overrides: dict[str, AgentConfig] = getattr(config, "agent_config_overrides", {})
    agents = [reg.cls(config, overrides.get(name, reg.config)) for name, reg in _REGISTRY.items()]
    # Logs: "instantiated N agent(s): ['arxiv', 's2', ...]"
    return agents
```

**Usage:**

```python
config = ResearchConfig(
    agent_config_overrides={
        "s2": AgentConfig(rate_limit_rps=0.5),  # Override S2 rate limit
    }
)
agents = get_agents(config)  # S2 gets 0.5 RPS; others use registry defaults
```

---

### get_registry_summary() → str

**Return a formatted roster of registered agents for use in LLM prompts.**

```python
def get_registry_summary() -> str:
    """Return a formatted multi-line string of the agent roster.

    Format per agent:
        [name] description
          Capabilities: ...
          Limitations: ...
          Example queries: ...

    Returns placeholder string when no agents are registered.
    """
```

**Example Output:**

```text
[arxiv] Retrieves preprints and papers from ArXiv covering physics, math, CS, and quantitative biology.
  Capabilities: Full LaTeX source for most CS/physics/math papers; PDF fallback when LaTeX is unavailable
  Limitations: No clinical or biomedical content (use pubmed); Preprints may not be peer-reviewed
  Example queries: "quantum error correction with surface codes"; "transformer attention mechanism efficiency improvements"

[s2] Retrieves citation-rich paper snippets from Semantic Scholar spanning all academic disciplines.
  ...
```

**Usage:**

- Injected into the query planner's system prompt
- Enables the LLM to route sub-queries intelligently

---

### \_REGISTRY

```python
_REGISTRY: dict[str, AgentRegistration] = {}
```

**Global registry dict. Keyed by agent name string.**

Access via `get_agents()` and `get_registry_summary()` — do not access directly unless implementing registry introspection.

---

## Cache Strategies

The `mara/agents/cache.py` module provides pluggable search result caching.

### SearchCache Protocol

```python
@runtime_checkable
class SearchCache(Protocol):
    """Async key-value cache keyed on (agent_type, query)."""

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        """Return cached chunks or None on a miss."""
        ...

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        """Store chunks under (agent_type, query)."""
        ...
```

**Design:**

- Async interface so it can be dropped in without blocking
- Keyed on `(agent_type, query)` tuple (e.g., `("s2", "quantum computing")`)
- `@runtime_checkable` allows pydantic to validate instances with `isinstance()`

**Integration in `_fetch_with_retry()`:**

1. Cache check (hit skips rate limiting and network call)
2. Rate-limit acquisition
3. Network call (if cache miss)
4. Cache population (on success only)

---

### NoOpCache

Default cache. All lookups miss; writes are discarded.

```python
class NoOpCache:
    """Default cache — all lookups miss, writes are discarded."""

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        return None

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        pass
```

**Zero overhead** when caching is not needed.

---

### InMemoryCache

Simple dict-backed cache for the lifetime of one pipeline run.

```python
class InMemoryCache:
    """Per-pipeline-run dict cache.

    Useful when the query planner emits overlapping sub-queries directed at
    the same agent. Stored for the lifetime of the InMemoryCache instance.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[RawChunk]] = {}

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        return self._store.get((agent_type, query))

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        self._store[(agent_type, query)] = chunks
```

**Usage:**

```python
config = ResearchConfig(search_cache=InMemoryCache())
```

**Use Cases:**

- Interactive research shells where the same query is executed multiple times
- Multi-run comparisons in tests

---

### Implementing a Custom Cache

Any class implementing the `SearchCache` protocol can be used. Store cached results in JSON or other safe formats:

```python
class FileSystemCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    async def get(self, agent_type: str, query: str) -> list[RawChunk] | None:
        cache_file = self._get_cache_file(agent_type, query)
        if cache_file.exists():
            import json
            data = json.loads(cache_file.read_text())
            return [RawChunk(**item) for item in data]
        return None

    async def set(self, agent_type: str, query: str, chunks: list[RawChunk]) -> None:
        import json
        cache_file = self._get_cache_file(agent_type, query)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(chunk) for chunk in chunks]
        cache_file.write_text(json.dumps(data))

    def _get_cache_file(self, agent_type: str, query: str):
        import hashlib
        key_hash = hashlib.sha256(f"{agent_type}:{query}".encode()).hexdigest()
        return Path(self.cache_dir) / f"{key_hash}.json"
```

Then pass it to config:

```python
config = ResearchConfig(search_cache=FileSystemCache("/tmp/mara_cache"))
```

---

## Chunk Filtering

The `mara/agents/filtering.py` module provides pluggable filtering strategies applied after chunking.

### ChunkFilter Protocol

```python
@runtime_checkable
class ChunkFilter(Protocol):
    """Strategy interface for post-chunking filtering.

    Receive the full chunk list and original query; return the filtered subset.
    Marked @runtime_checkable so pydantic can use isinstance.
    """

    def filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]: ...
```

**Integration in `_filter()`:**

```python
def _filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
    chunk_filter: ChunkFilter = self.config.chunk_filter
    return chunk_filter.filter(chunks, query)
```

---

### CapFilter

Hard-cap filter. Limits chunks per URL and globally per agent.

```python
@dataclass
class CapFilter:
    """Hard-cap filter: at most max_chunks_per_url per URL,
    then at most max_chunks_per_agent globally.

    Order is preserved within each URL; per-URL cap applied first,
    then global cap truncates the tail.
    """

    max_chunks_per_url: int = 3
    max_chunks_per_agent: int = 50

    def filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
        """Apply per-URL cap, then global cap."""
```

**Example:**

- 200 chunks total
- 50 chunks from arxiv.org
- 100 chunks from github.com
- 50 chunks from springer.com

With `CapFilter(max_chunks_per_url=10, max_chunks_per_agent=30)`:

1. Per-URL: arxiv.org→10, github.com→10, springer.com→10 (30 total)
2. Global: keep first 30 (truncates springer from 10 to 10, which is already there)

Result: 10 + 10 + 10 = 30 chunks

---

### EmbeddingFilter

Stub — not yet implemented.

```python
@dataclass
class EmbeddingFilter:
    """Stub — not yet implemented.

    Will rank chunks by cosine similarity to the query embedding and keep
    the top max_chunks_per_agent results above similarity_threshold.
    """

    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.6
    max_chunks_per_agent: int = 50

    def filter(self, chunks: list[RawChunk], query: str) -> list[RawChunk]:
        raise NotImplementedError("EmbeddingFilter is not yet implemented")
```

---

## Specialist Agents

All specialist agents extend `SpecialistAgent` and are registered via the `@agent()` decorator.

### ArXiv Agent

**Module:** `mara/agents/arxiv/`

**Discovery:** ArXiv Atom API (`export.arxiv.org/api/query`)

**Content Strategy:** Four-level fallback chain per paper:

1. **LaTeX source** (`source_type="latex"`) — Extracted from source tarball
2. **PDF from tarball** (`source_type="pdf_from_tarball"`) — Embedded in tarball
3. **Rendered PDF** (`source_type="pdf_rendered"`) — Downloaded from arxiv.org/pdf/
4. **Abstract** (`source_type="abstract_only"`) — From the Atom feed

**Rate Limiting:**

- `_rate_limit_interval = 3.0` (class variable) — 3 seconds between discovery calls
- Additional 3-second sleep **inside `_search()`** between per-paper content fetches
- Respects ArXiv's API limits

**Chunking Override:**

- LaTeX sections are pre-chunked and passed through unchanged
- PDF and abstract chunks use the sliding-window splitter

**Key Files:**

- `agent.py` — Main agent; Atom feed parsing; fallback logic
- `fetcher.py` — HTTP fetching; tarball extraction; PDF text extraction
- `latex_parser.py` — LaTeX source parsing; section extraction; text cleaning

**Source Types:**

```python
LATEX = "latex"
PDF_FROM_TARBALL = "pdf_from_tarball"
PDF_RENDERED = "pdf_rendered"
ABSTRACT_ONLY = "abstract_only"
```

**Key Features:**

- Colon in `all:<term>` queries is NOT percent-encoded (manual query string construction)
- Graceful fallback through all four levels
- Comprehensive LaTeX to plain-text conversion via `pylatexenc` with regex fallback

---

### Semantic Scholar Agent

**Module:** `mara/agents/semantic_scholar/`

**Discovery:** Semantic Scholar Snippet Search API (`api.semanticscholar.org/graph/v1/snippet/search`)

**Content Strategy:** Pre-extracted snippets (one per result item). No full-text access.

**Rate Limiting:**

- `_get_rate_limit_interval()` returns `1.0 / config.s2_max_rps` (≤1 RPS by default)
- Configurable via `ResearchConfig.s2_max_rps`

**Chunking Override:**

- Snippets are pre-chunked; passed through unchanged

**Source Types:**

```python
SNIPPET = "snippet"
```

**Configuration Parameters:**

- `s2_api_key` — Required; obtain from Semantic Scholar API
- `s2_max_rps` — Max requests per second (default 1)
- `s2_max_results` — Max results per query (default configurable)

**Key Features:**

- High signal-to-noise ratio (pre-filtered snippets)
- Cross-disciplinary coverage
- Citation metadata useful for literature review

---

### PubMed Agent

**Module:** `mara/agents/pubmed/`

**Discovery:** NCBI eUtils — Sequential chain of three API calls per paper:

1. `esearch.fcgi` — Get PMIDs for the query
2. `esummary.fcgi` — Fetch metadata (title, PMC ID if available)
3. `efetch.fcgi` — Get full text (PMC XML) or abstract

**Content Strategy:** Two-level fallback per paper:

1. **PMC full-text** (`source_type="pmc_xml"`) — Parsed `<sec>` XML sections
2. **Abstract** (`source_type="abstract_only"`) — `<AbstractText>` elements

**Rate Limiting:**

- `_get_rate_limit_interval()` returns `1.0 / config.pubmed_rate_limit_per_second`
- Default 3 req/s without API key; 10 req/s with key
- Additional **per-eUtils-call delay** via `asyncio.sleep()` inside `_search()` for intra-request pacing

**Chunking Override:**

- PMC sections and abstracts are pre-chunked; passed through unchanged

**Source Types:**

```python
PMC_XML = "pmc_xml"
ABSTRACT_ONLY = "abstract_only"
```

**Configuration Parameters:**

- `pubmed_rate_limit_per_second` — Rate limiting (default 3)
- `ncbi_api_key` — Optional; raises limit to 10 req/s
- `pubmed_max_results` — Max results per query

**Key Features:**

- `_ncbi_params()` helper omits `api_key` when blank (NCBI returns 400 otherwise)
- Section-level parsing preserves document structure
- Strong coverage of biomedical/life sciences

**Helper Modules:**

- `search.py` — `parse_article_metadata()` — extracts pmid, title, pmc_id from esummary results
- `fulltext.py` — `parse_pmc_sections()`, `parse_abstract_xml()` — XML parsing helpers

---

### CORE Agent

**Module:** `mara/agents/core/`

**Discovery:** CORE API v3 (`api.core.ac.uk/v3/search/works`)

**Content Strategy:** Three-level fallback per paper:

1. **Full text** (`source_type="fulltext"`) — `fullText` field from API
2. **PDF download** (`source_type="pdf_downloaded"`) — Fetch from `downloadUrl`, extract with pypdf
3. **Abstract** (`source_type="abstract_only"`) — `abstract` field from API

**Rate Limiting:**

- No class-level override (uses base default 0.0, i.e., no rate limit)
- CORE API itself has per-call limits; no need for client-side throttling

**Chunking Override:**

- CORE content is pre-chunked; passed through unchanged

**Source Types:**

```python
FULLTEXT = "fulltext"
PDF_DOWNLOADED = "pdf_downloaded"
ABSTRACT_ONLY = "abstract_only"
```

**Configuration Parameters:**

- `core_api_key` — Required; Bearer token for CORE API
- `core_max_results` — Max results per query

**Key Features:**

- Large corpus of open-access papers across all fields
- PDF fallback when full-text field is absent
- Covers education, social sciences, interdisciplinary research

**Helper Module:**

- `fetcher.py` — `extract_pdf_text()` — PDF text extraction via pypdf (shared with arxiv)

---

### Web Agent

**Module:** `mara/agents/web/`

**Discovery:** Brave Search API (`api.search.brave.com/res/v1/web/search`)

**Filtering:** Two-pass process:

1. **Domain-tier regex** (synchronous) — BLOCK/DEFAULT/GOOD/PRIORITY classification
2. **Optional LLM ranking** (async) — Relevance filtering using the configured LLM

**Content Strategy:**

- Extracts URLs from Brave Search (web, news, discussions, faq sections)
- Filters URLs by domain tier
- Optionally ranks with LLM
- Scrapes each URL with Firecrawl
- Returns markdown text from each page

**Rate Limiting:**

- No class-level override (uses base default 0.0)

**Chunking Override:**

- Web content is pre-chunked; passed through unchanged

**Source Types:**

```python
WEB = "web"
```

**Configuration Parameters:**

- `brave_api_key` — Required; Brave Search API token
- `web_max_results` — Max results from Brave Search
- `web_timeout_seconds` — HTTP timeout
- `web_max_scrape_urls` — Max URLs to actually scrape (caps Brave results)
- `web_llm_url_ranking` — Enable/disable LLM relevance ranking (default True)
- `firecrawl_api_key` — Required; Firecrawl scraping service
- `scrape_timeout_seconds` — Firecrawl timeout per URL
- `brave_freshness` — Optional; e.g., "pd" (past day), "pw" (past week)

**Domain Tiers:**

| Tier         | Examples                                                       |
| ------------ | -------------------------------------------------------------- |
| **BLOCK**    | facebook, twitter, instagram, tiktok, reddit, youtube, etc.    |
| **PRIORITY** | .edu, .gov, .ac.\* domains; nature, springer, ieee, acm, etc.  |
| **GOOD**     | wikipedia, bbc, reuters, nytimes, bbc, techcrunch, wired, etc. |
| **DEFAULT**  | Everything else                                                |

**Process:**

1. Brave Search discovers URLs across multiple sections (web, news, discussions, faq)
2. Duplicates across sections are dropped
3. BLOCK-tier URLs are removed; survivors sorted PRIORITY → GOOD → DEFAULT
4. If `web_llm_url_ranking=True`, the LLM re-ranks survivors by relevance to the query
5. Top `web_max_scrape_urls` are scraped with Firecrawl
6. Each page's markdown is wrapped in a RawChunk

**Helper Modules:**

- `scraper.py` — `scrape_url()` — Firecrawl integration via `asyncio.to_thread`
- `url_filter.py` — `filter_urls_by_tier()`, `rank_urls_with_llm()`, `DomainTier` enum

**Key Features:**

- Live web content and current events
- Grey literature (blogs, industry reports, documentation)
- Slow due to scraping overhead
- Quality varies by domain

---

## Adding a New Agent

To add a new specialist agent, follow this step-by-step guide.

### Step 1: Create the Agent Module

Create a directory under `mara/agents/`:

```bash
mkdir mara/agents/mynewagent
touch mara/agents/mynewagent/__init__.py
touch mara/agents/mynewagent/agent.py
```

### Step 2: Implement the Agent Class

In `mara/agents/mynewagent/agent.py`:

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from mara.agents.base import SpecialistAgent
from mara.agents.registry import agent
from mara.agents.types import RawChunk, SubQuery

_log = logging.getLogger(__name__)

# Source type constants
MY_SOURCE_TYPE = "my_source_type"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@agent(
    "mynewagent",
    description="Brief one-sentence summary of what this agent retrieves.",
    capabilities=[
        "Capability 1",
        "Capability 2",
        "Capability 3",
    ],
    limitations=[
        "Limitation 1",
        "Limitation 2",
    ],
    example_queries=[
        "Example query 1",
        "Example query 2",
    ],
)
class MyNewAgent(SpecialistAgent):
    """Retrieves research from [source]."""

    # Optional: class-level rate limiting (seconds between calls)
    _rate_limit_interval: float = 0.0  # or override _get_rate_limit_interval()

    # Optional: override _chunk() if content is pre-chunked
    def _chunk(self, raw: list[RawChunk]) -> list[RawChunk]:
        """If content is pre-chunked, pass through unchanged."""
        return raw

    async def _search(self, sub_query: SubQuery) -> list[RawChunk]:
        """Fetch and return raw chunks for *sub_query*.

        Do NOT hash, retry, or sleep — that's all handled by the base class.
        Just fetch and return RawChunk list.
        """
        retrieved_at = _now_iso()

        async with httpx.AsyncClient() as client:
            # Fetch from your API
            response = await client.get(
                "https://api.example.com/search",
                params={"q": sub_query.query},
            )
            response.raise_for_status()

        chunks: list[RawChunk] = []
        for item in response.json().get("results", []):
            chunks.append(
                RawChunk(
                    url=item.get("url"),
                    text=item.get("content"),
                    retrieved_at=retrieved_at,
                    source_type=MY_SOURCE_TYPE,
                    sub_query=sub_query.query,
                )
            )

        return chunks
```

### Step 3: Add Configuration Parameters (if needed)

Edit `mara/config.py` and add any new parameters to `ResearchConfig`:

```python
class ResearchConfig(BaseSettings):
    ...
    mynewagent_api_key: str = Field(default="", description="API key for mynewagent")
    mynewagent_max_results: int = Field(default=50, description="Max results per query")
    mynewagent_rate_limit: float = Field(default=1.0, description="Requests per second")
    ...
```

### Step 4: Update `mara/agents/__init__.py` (if needed)

If using helper modules, add imports:

```python
# At the end of the file to avoid circular imports
from mara.agents.mynewagent import MyNewAgent  # noqa: F401
```

### Step 5: Add Environment Variable

Update the required environment variables section in `CLAUDE.md`:

```bash
MYNEWAGENT_API_KEY=...  # mynewagent discovery
```

### Step 6: Write Tests

Create `tests/agents/mynewagent/` with:

```python
import pytest
from mara.agents.mynewagent.agent import MyNewAgent
from mara.agents.types import SubQuery
from mara.config import ResearchConfig


@pytest.fixture
def agent():
    config = ResearchConfig(
        mynewagent_api_key="test_key",
    )
    return MyNewAgent(config)


@pytest.fixture
def reset_rate_limits():
    yield
    MyNewAgent._reset_rate_limit_state()


@pytest.mark.asyncio
async def test_search_returns_chunks(agent, httpx_mock):
    httpx_mock.add_response(
        method="GET",
        json={
            "results": [
                {
                    "url": "https://example.com/1",
                    "content": "First result",
                },
                {
                    "url": "https://example.com/2",
                    "content": "Second result",
                },
            ]
        },
    )

    sub_query = SubQuery("machine learning")
    chunks = await agent._search(sub_query)

    assert len(chunks) == 2
    assert chunks[0].url == "https://example.com/1"
    assert chunks[0].text == "First result"
```

### Step 7: Test Registration

Verify the agent is registered:

```python
from mara.agents.registry import _REGISTRY, get_registry_summary

def test_agent_registered():
    assert "mynewagent" in _REGISTRY
    summary = get_registry_summary()
    assert "[mynewagent]" in summary
```

### Step 8: Run the Pipeline

The agent is now automatically available in the pipeline:

```python
from mara.agent.graph import run_research
from mara.config import ResearchConfig

config = ResearchConfig(mynewagent_api_key="...")
report = await run_research("your query", config)
```

The query planner will see `mynewagent` in `get_registry_summary()` and can route sub-queries to it.

---

## Integration Points

The agents package integrates with the rest of MARA via:

1. **`mara/agent/graph.py`** — LangGraph pipeline consumes `get_agents()` and `get_registry_summary()`
2. **`mara/config.py`** — `ResearchConfig` holds all tunable parameters
3. **`mara/merkle/`** — Hashing and tree building consumed by `SpecialistAgent.run()`
4. **`mara/llm.py`** — `make_llm()` used by web agent's URL ranking

---

## Testing Guidelines

### Unit Tests

Test individual agent methods in isolation:

```python
def test_chunk_splits_long_text(agent):
    raw = [RawChunk(url="...", text="x" * 10000, ...)]
    chunks = agent._chunk(raw)
    assert len(chunks) > 1
```

### Integration Tests

Test the full `run()` pipeline with mocked HTTP:

```python
@pytest.mark.asyncio
async def test_run_hashes_chunks(agent, httpx_mock):
    httpx_mock.add_response(json={"results": [...]})
    findings = await agent.run(SubQuery("query"))
    assert findings.chunk_count > 0
    assert findings.merkle_root != ""
```

### Mocking

Mock at the `httpx.AsyncClient.get` level, not internal helpers:

```python
httpx_mock.add_response(
    method="GET",
    url="https://api.example.com/search",
    json={"results": [...]},
)
```

### Coverage

Aim for 98%+ coverage. Use `pytest --cov=mara.agents`:

```bash
uv run pytest --cov=mara.agents --cov-branch --cov-report=term-missing
```

---

## Best Practices

1. **Immutable Data** — Never mutate `RawChunk` or `VerifiedChunk`; use dataclass replacement when needed
2. **Logging** — Always include `extra={"agent": self._agent_type()}` in log calls
3. **Error Handling** — Let `_fetch_with_retry()` handle HTTP errors; don't catch in `_search()`
4. **Rate Limiting** — Set `_rate_limit_interval` or override `_get_rate_limit_interval()` early
5. **Pre-chunking** — If content is already chunked, override `_chunk()` to pass through
6. **Source Types** — Use meaningful constants, not magic strings
7. **Testing** — Reset rate limits in teardown: `MyAgent._reset_rate_limit_state()`
8. **Configuration** — Always use `self.config` for parameters, never hardcode defaults
9. **Timestamps** — Use `datetime.now(timezone.utc).isoformat()` for UTC ISO-8601 strings
10. **Async/Await** — Use `asyncio` for concurrency; call Firecrawl via `asyncio.to_thread`

---

## Glossary

- **Agent** — A specialist that retrieves research from one source (arxiv, pubmed, etc.)
- **Chunk** — A segment of source text with metadata (URL, timestamp, source type)
- **RawChunk** — Unverified chunk before hashing
- **VerifiedChunk** — Chunk with cryptographic hash; produced by `SpecialistAgent.run()`
- **AgentFindings** — A collection of verified chunks plus their Merkle tree (output of one agent)
- **CertifiedReport** — Final output with report text, forest tree, and all chunks
- **Merkle Tree** — Binary tree where each leaf is a chunk hash; enables proof of chunk membership
- **Forest Tree** — Meta-tree of all agent sub-trees; enables two-level proofs
- **Sub-query** — A decomposed research question routed to one or more agents
- **Discovery** — Finding relevant sources (e.g., ArXiv API search)
- **Fallback** — Trying alternative content strategies when preferred one fails (e.g., LaTeX → PDF → abstract)
- **Rate Limiting** — Enforcing minimum delay between API calls to respect service limits
- **Retry** — Retrying transient failures (429, 5xx) with exponential backoff

---

---

*End of documentation for `mara/agents/`.*
