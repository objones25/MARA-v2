# MARA — Multi-Agent Research Assistant

MARA runs a parallel research pipeline across academic and web sources, synthesises a structured report, and returns it together with a cryptographic provenance chain so every claim can be traced back to its source.

## How it works

```
query_planner → [route_to_agents] → run_agent (×N) → corpus_assembler → report_synthesizer → certified_output
```

1. **Query planner** — an LLM decomposes the user's question into targeted sub-queries and assigns each to the most appropriate agent.
2. **Fan-out** — LangGraph's `Send()` API dispatches sub-queries in parallel.
3. **Specialist agents** — each agent fetches source text, hashes every chunk immediately, and assembles the hashes into a Merkle sub-tree.
4. **Corpus assembler** — merges all sub-tree roots into a single `ForestTree`, assigns globally unique chunk indices, and builds the two-level proof structure.
5. **Report synthesiser** — an LLM reads the indexed corpus and writes a report with inline citations (`[ML:index:hash]`).
6. **Certified output** — resolves citations to source URLs, appends a `## References` section, and packages everything into an immutable `CertifiedReport`.

## Agents

| Agent    | Sources                                                    |
| -------- | ---------------------------------------------------------- |
| `arxiv`  | arXiv export API — LaTeX source, rendered PDF, or abstract |
| `s2`     | Semantic Scholar snippet search                            |
| `pubmed` | NCBI PubMed/PMC full text and abstracts                    |
| `core`   | CORE API v3 full text, PDF download, or abstract           |
| `web`    | Brave Search + Firecrawl scraping                          |

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/objones25/MARA-v2.git
cd MARA-v2
uv sync
```

## Configuration

Copy `.env.example` to `.env` (or export the variables directly):

```bash
BRAVE_API_KEY=...
HF_TOKEN=...
FIRECRAWL_API_KEY=...
CORE_API_KEY=...
S2_API_KEY=...
NCBI_API_KEY=...
```

Optional variables and their defaults:

| Variable                       | Default                            | Description                                       |
| ------------------------------ | ---------------------------------- | ------------------------------------------------- |
| `LOG_LEVEL`                    | `INFO`                             | Python log level (`DEBUG`, `INFO`, `WARNING`, …)  |
| `DEFAULT_MODEL`                | `Qwen/Qwen3-30B-A3B-Instruct-2507` | HuggingFace model ID                              |
| `LLM_PROVIDER`                 | `featherless-ai`                   | LangChain provider string                         |
| `MAX_RETRIES`                  | `3`                                | HTTP retry attempts per agent call                |
| `RETRY_BACKOFF_BASE`           | `2.0`                              | Exponential back-off base (seconds)               |
| `S2_MAX_RPS`                   | `1.0`                              | Semantic Scholar requests per second              |
| `PUBMED_RATE_LIMIT_PER_SECOND` | `3.0`                              | PubMed requests per second (10 with NCBI API key) |
| `CHUNK_SIZE`                   | `1000`                             | Character window size for chunking                |
| `CHUNK_OVERLAP`                | `200`                              | Character overlap between adjacent chunks         |

## Usage

```bash
# Plain text report
uv run mara "what are the latest advances in quantum error correction?"

# JSON output (includes forest root and chunk count)
uv run mara --json "what are the latest advances in quantum error correction?"

# Verbose logging
LOG_LEVEL=DEBUG uv run mara "your question"
```

## Testing

```bash
# Run all tests with coverage
uv run pytest

# Single file
uv run pytest tests/agents/test_base.py -v
```

Coverage is enforced at ≥ 98% (branch coverage) via `pytest.ini`.

## Cryptographic provenance

Every chunk carries a SHA-256 hash of its `(url, text, retrieved_at)` triple. Hashes are assembled into a binary Merkle tree per agent; agent roots become leaves of a `ForestTree`. Any chunk can be independently verified with a two-level Merkle proof without re-running the pipeline.

```python
from mara.merkle.proof import generate_proof, verify_proof

proof = generate_proof(findings.merkle_tree, chunk_index=3)
assert verify_proof(chunk.hash, proof, findings.merkle_root, "sha256")
```

## Adding a new agent

1. Create `mara/agents/<name>/agent.py` with a class that extends `SpecialistAgent`.
2. Implement `_search(sub_query) -> list[RawChunk]` — pure HTTP fetch, no retries or locks.
3. Apply `@agent("name", description=..., capabilities=[...])` to the class.
4. Import the module in `mara/cli/run.py` so the decorator runs at startup.

The base class provides rate limiting, retry with exponential back-off, caching, hashing, and Merkle tree assembly automatically.
