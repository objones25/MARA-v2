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

### Key Types (`mara/agents/types.py`)

- **`VerifiedChunk`** — frozen dataclass: `hash`, `url`, `text`, `retrieved_at` (ISO-8601 UTC), `source_type`, `sub_query`, `chunk_index`. Hash is SHA-256 of `canonical_serialise(url, text, retrieved_at)`.
- **`AgentFindings`** — frozen dataclass: `agent_type`, `query`, `chunks`, `merkle_root`, `merkle_tree`. Merkle root is recomputed and verified in `__post_init__`.

### Agent System

Agents self-register via `@agent("name")` decorator in `mara/agents/registry.py`. Adding a new agent means writing the class and applying the decorator — nothing else changes.

All agents extend `SpecialistAgent` (`mara/agents/base.py`). The base `run()` method handles hashing and Merkle tree construction; subclasses only implement `_retrieve(sub_query) -> list[RawChunk]`.

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

### Merkle Layer (`mara/merkle/`)

- `hasher.py` — `canonical_serialise`, `hash_chunk`
- `tree.py` — `build_merkle_tree`, `MerkleTree`
- `proof.py` — chunk proof generation/verification
- `forest.py` — `ForestTree`, `build_forest_tree` (meta-tree over agent sub-trees)

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
