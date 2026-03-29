# MARA Agent Pipeline Documentation

**Last Updated:** 2026-03-29

The `mara/agent/` package implements a LangGraph-based research pipeline that orchestrates multi-agent query decomposition, parallel information retrieval, and report synthesis with cryptographic provenance.

## Table of Contents

1. [Package Overview](#package-overview)
2. [State Management](#state-management)
3. [Graph Architecture](#graph-architecture)
4. [Routing Strategy](#routing-strategy)
5. [Pipeline Nodes](#pipeline-nodes)
   - [Query Planner Node](#query-planner-node)
   - [Run Agent Node](#run-agent-node)
   - [Corpus Assembler Node](#corpus-assembler-node)
   - [Chunk Selector Node](#chunk-selector-node)
   - [Report Synthesizer Node](#report-synthesizer-node)
   - [Certified Output Node](#certified-output-node)
6. [Configuration Integration](#configuration-integration)
7. [Error Handling](#error-handling)
8. [Usage Examples](#usage-examples)
9. [Build Order and Import Dependencies](#build-order-and-import-dependencies)

---

## Package Overview

### Purpose

The agent pipeline transforms a user's research query into a cryptographically verified report by:

1. Decomposing the query into focused sub-queries
2. Routing sub-queries to specialized agents (arXiv, Semantic Scholar, PubMed, CORE, web)
3. Executing agents in parallel
4. Assembling findings into a unified Merkle forest
5. Synthesizing a narrative report with inline citations
6. Producing a final `CertifiedReport` with full provenance

### Core Invariant

**Cryptographic verification happens during construction, not after.** The `AgentFindings.__post_init__` method recomputes the Merkle root from chunks and raises `ValueError("merkle_root mismatch")` if invalid. A successfully constructed `CertifiedReport` is guaranteed authentic.

### Data Flow Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│ START                                                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ query_planner_node                                                  │
│ ─────────────────                                                   │
│ • Reads: original_query                                             │
│ • Calls LLM with agent roster to decompose query                    │
│ • Parses JSON array into SubQuery list (or fallback)                │
│ • Outputs: sub_queries                                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ route_to_agents (conditional edge)                                  │
│ ─────────────────────────────────────                               │
│ Per sub-query:                                                      │
│ • If sub_query.agent names a registered agent → Send to that agent  │
│ • Else → Send to every registered agent (broadcast fallback)        │
│ • If no sub-queries or no agents → bypass to corpus_assembler      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
      ┌──────────────────────────┐  ┌──────────────────────────┐
      │ run_agent (×N×M)         │  │ run_agent (×N×M)         │
      │ ───────────────────────  │  │ ───────────────────────  │
      │ • Lookup agent class     │  │ • Lookup agent class     │
      │ • Instantiate + config   │  │ • Instantiate + config   │
      │ • Run agent.run()        │  │ • Run agent.run()        │
      │ • Return: findings       │  │ • Return: findings       │
      └──────────────────────────┘  └──────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    (operator.add reducer merges findings lists)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ corpus_assembler_node                                               │
│ ──────────────────────                                              │
│ • Group chunks by agent_type                                        │
│ • Sort deterministically                                            │
│ • Build MerkleTree per agent                                        │
│ • Build ForestTree from sub-tree roots                              │
│ • Assign global chunk indices                                       │
│ • Outputs: forest_tree, flattened_chunks                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ chunk_selector_node                                                 │
│ ────────────────────                                                │
│ • Deduplicate chunks by hash                                        │
│ • BM25Plus-rank chunks against original_query                       │
│ • Cap at chunk_selector_cap (default 50)                            │
│ • Outputs: selected_chunks                                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ report_synthesizer_node                                             │
│ ──────────────────────────                                          │
│ • Format chunks as [index:short_hash] text                          │
│ • Call LLM to synthesize report with inline citations               │
│ • Outputs: report                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ certified_output_node                                               │
│ ──────────────────────                                              │
│ • Parse citation indices from report                                │
│ • Build References section                                          │
│ • Construct CertifiedReport                                         │
│ • Outputs: certified_report                                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ END                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## State Management

### `GraphState` (TypedDict, total=False)

The immutable state dict threaded through all nodes. `total=False` makes every key optional so nodes return only the fields they update.

```python
class GraphState(TypedDict, total=False):
    original_query: str
    """The user's research question (set by run_research)."""

    sub_queries: list[SubQuery]
    """Decomposed sub-queries with agent assignments (set by query_planner)."""

    findings: Annotated[list[AgentFindings], operator.add]
    """Accumulated findings from all agents.

    The operator.add reducer enables merging: each run_agent node returns
    {"findings": [one_AgentFindings]}, and LangGraph accumulates them into
    a single list before corpus_assembler runs.
    """

    forest_tree: ForestTree
    """Two-level Merkle forest of agent sub-trees (set by corpus_assembler)."""

    flattened_chunks: list[VerifiedChunk]
    """All chunks in sorted agent order with global indices
    (set by corpus_assembler)."""

    selected_chunks: list[VerifiedChunk]
    """Deduplicated, BM25-ranked subset of flattened_chunks, capped at
    chunk_selector_cap. Preferred by report_synthesizer over flattened_chunks.
    (set by chunk_selector)."""

    report: str
    """The narrative research report with inline citations
    (set by report_synthesizer)."""

    certified_report: CertifiedReport
    """The final output: query + report + provenance
    (set by certified_output)."""
```

### `AgentRunState` (TypedDict)

The payload sent to each parallel `run_agent` invocation via `Send()`.

```python
class AgentRunState(TypedDict):
    sub_query: SubQuery
    """The sub-query to execute."""

    agent_type: str
    """The registered agent name (e.g., "pubmed", "s2")."""
```

---

## Graph Architecture

### `build_graph()` Function

```python
def build_graph() -> StateGraph:
    """Compile and return the MARA research pipeline graph.

    Graph topology::

        START → query_planner → [route_to_agents] → run_agent (×N×M)
              → corpus_assembler → chunk_selector → report_synthesizer → certified_output → END
    """
```

**Topology Notes:**

- **Linear baseline:** START → query_planner → route_to_agents → corpus_assembler → chunk_selector → report_synthesizer → certified_output → END
- **Parallel fan-out:** route_to_agents emits multiple `Send("run_agent", ...)` invocations
- **Parallel fan-in:** LangGraph merges all `run_agent` results using the `operator.add` reducer on `GraphState.findings`
- **Direct bypass:** When route_to_agents has no sub-queries or agents, it returns the string `"corpus_assembler"` (a bypass edge), skipping the run_agent node entirely

**Edge Configuration:**

| From               | To                           | Type        | Notes                               |
| ------------------ | ---------------------------- | ----------- | ----------------------------------- |
| START              | query_planner                | Direct      |                                     |
| query_planner      | run_agent / corpus_assembler | Conditional | Decided by route_to_agents function |
| run_agent          | corpus_assembler             | Direct      | All parallel invocations converge   |
| corpus_assembler   | chunk_selector               | Direct      |                                     |
| chunk_selector     | report_synthesizer           | Direct      |                                     |
| report_synthesizer | certified_output             | Direct      |                                     |
| certified_output   | END                          | Direct      |                                     |

### `run_research()` Function

```python
async def run_research(query: str, config: ResearchConfig) -> CertifiedReport:
    """Run the full MARA pipeline for query and return a CertifiedReport.

    Args:
        query:  The user's research question.
        config: Fully-populated ResearchConfig (all API keys required).

    Returns:
        A CertifiedReport containing the narrative report and its full
        Merkle provenance chain.
    """
```

**Execution Flow:**

1. Calls `build_graph()` to compile the StateGraph
2. Invokes the graph with `ainvoke()`:
   - Initial state: `{"original_query": query}`
   - Config: `{"configurable": {"research_config": config}}`
3. Retrieves `result["certified_report"]` and returns it

**Important:** Agent modules (e.g., `mara.agents.arxiv`, `mara.agents.pubmed`) **must be imported before calling `run_research()`** so their `@agent()` decorators populate the `_REGISTRY`. The CLI does this in its `run()` function.

---

## Routing Strategy

### `route_to_agents()` Function

```python
def route_to_agents(state: GraphState, config: RunnableConfig) -> list[Send] | str:
    """Return Send("run_agent", ...) objects for each sub-query.

    Routing strategy (per sub-query):
    - **Directed**: if sub_query.agent names a registered agent, send to
      that agent only.
    - **Broadcast fallback**: if sub_query.agent is empty or unrecognized,
      send to every registered agent.

    Returns "corpus_assembler" (a bypass edge) when the sub-query list or
    agent registry is empty.
    """
```

**Routing Logic:**

```python
sends: list[Send] = []
for sq in sub_queries:
    if sq.agent and sq.agent in _REGISTRY:
        # Directed: send to the named agent only
        sends.append(Send("run_agent", {"sub_query": sq, "agent_type": sq.agent}))
    else:
        # Broadcast fallback: send to every registered agent
        for at in agent_types:
            sends.append(Send("run_agent", {"sub_query": sq, "agent_type": at}))

return sends if sends else "corpus_assembler"
```

**Parallelism:**

- A query with 3 sub-queries routed to 1 agent each = 3 `run_agent` invocations (serial or parallel depending on graph executor)
- A query with 3 sub-queries all broadcast to 5 agents = 15 `run_agent` invocations
- Each invocation receives its own `AgentRunState` via `Send()`

**Bypass Edge:**

If `sends` is empty (no sub-queries or no agents in registry), the function returns the string `"corpus_assembler"`, which LangGraph treats as a direct edge bypass. This prevents `run_agent` from running and proceeds directly to corpus assembly.

---

## Pipeline Nodes

### Query Planner Node

**File:** `mara/agent/nodes/query_planner.py`

**Purpose:** Decompose the original query into focused sub-queries and assign each to the most appropriate agent.

**Signature:**

```python
async def query_planner_node(state: GraphState, config: RunnableConfig) -> dict:
    """Decompose original_query into a list of SubQuery objects.

    Calls the LLM with a decomposition prompt that includes the live agent
    roster so the LLM can assign each sub-query to the most appropriate agent.
    Falls back to a single SubQuery wrapping the original query if the
    LLM response cannot be parsed or yields no valid sub-queries.
    """
```

**Algorithm:**

1. **Extract config:** Retrieve `research_config` from `config["configurable"]["research_config"]`
2. **Build system prompt:** Template + `get_registry_summary()` to inject the live agent roster
3. **Call LLM:** With system prompt + user query
4. **Parse response:** Use regex to find a JSON array in the LLM output
5. **Construct SubQuery objects:** From each array item (`{query, domain, agent}`)
6. **Fallback:** If parsing fails or no sub-queries result, return `[SubQuery(original_query)]`
7. **Return:** `{"sub_queries": sub_queries}`

**System Prompt Template:**

```text
You are a research query decomposition specialist.

Given a research query, decompose it into 3-5 focused sub-queries that together cover the topic comprehensively. For each sub-query, assign the single most appropriate agent from the roster below.

Routing constraints:
- Respect each agent's "Max sub-queries" limit shown in the roster; never exceed it for that agent
- Each sub-query must focus on a DISTINCT aspect of the research topic; do not generate two queries with overlapping scope
- Use concrete, searchable phrases (e.g. "machine learning classification performance", not "AI classification")
- For cross-domain topics, assign different agents to cover different angles (e.g. pubmed for clinical evidence, arxiv for mathematical foundations)

Available agents:
{agent_roster}

Return a JSON array of objects with "query", "domain", and "agent" fields.
- "query": the focused sub-question (non-empty string)
- "domain": a short topic hint, e.g. "empirical", "clinical", "theoretical" (may be empty)
- "agent": the agent name from the roster above that best fits this sub-query

Example:
[{"query": "...", "domain": "empirical", "agent": "pubmed"}, ...]

Return ONLY the JSON array, no other text.
```

**Parsing:**

- Regex: `r"\[.*\]"` (case-insensitive dotall mode) extracts the first JSON array
- For each item, creates a `SubQuery(query=item["query"], domain=item.get("domain", ""), agent=item.get("agent", ""))`
- Skips items with empty/whitespace-only `query` fields
- If the final list is empty or parsing failed, returns `[SubQuery(original_query)]`

**Logging:**

- DEBUG: `"query_planner produced {N} sub-queries for {query[:60]}"`
- WARNING: On parse failure (includes the exception)

---

### Run Agent Node

**File:** `mara/agent/nodes/run_agent.py`

**Purpose:** Execute a single registered agent against one sub-query.

**Signature:**

```python
async def run_agent_node(state: AgentRunState, config: RunnableConfig) -> dict:
    """Run one registered agent against one sub-query.

    Instantiates the agent class looked up by state["agent_type"] from the
    module-level _REGISTRY reference (patchable in tests). Returns
    {"findings": [AgentFindings]} so the operator.add reducer can accumulate
    results across all parallel invocations.
    """
```

**Algorithm:**

1. **Extract fields:** `agent_type` from state, `research_config` from LangGraph config
2. **Lookup agent class:** `agent_cls = _REGISTRY[agent_type].cls`
3. **Instantiate:** `agent = agent_cls(research_config)`
4. **Execute:** `findings = await agent.run(sub_query)`
5. **Error handling:**
   - On exception: logs WARNING and returns `{"findings": []}`
   - Does **not** re-raise (prevents one agent failure from crashing the pipeline)
6. **Return:** `{"findings": [findings]}` (list format for the reducer)

**Error Isolation:**

Each `run_agent` invocation is independent. If the PubMed agent fails, the arXiv agent still succeeds. The empty list from the failed agent is merged into the final `findings` list by the `operator.add` reducer.

**Logging:**

- DEBUG (entry): `"run_agent: agent={agent_type} query={sub_query.query[:60]}"`
- WARNING (on exception): `"run_agent: agent={agent_type} failed for query={query[:60]}: {exc}"`
- DEBUG (exit): `"run_agent: agent={agent_type} returned {N} chunk(s)"`

---

### Corpus Assembler Node

**File:** `mara/agent/nodes/corpus_assembler.py`

**Purpose:** Merge all agent findings into a unified Merkle forest and assign global chunk indices.

**Signature:**

```python
def corpus_assembler_node(state: GraphState, config: RunnableConfig) -> dict:
    """Merge all AgentFindings into a ForestTree and a flat chunk list.

    Multiple AgentFindings objects may share the same agent_type
    (one per sub-query × agent). This node:

    1. Groups chunks by agent_type.
    2. Sorts each group by (sub_query, chunk_index) for determinism.
    3. Builds one MerkleTree per agent from the merged chunk list.
    4. Calls build_forest_tree with one (agent_type, root) pair per agent.
    5. Assigns globally unique chunk_index values using dataclasses.replace.
    """
```

**Algorithm:**

1. **Group:** Iterate over all `AgentFindings`; group their chunks by `agent_type`
2. **Sort:** Within each agent group, sort by `(sub_query, chunk_index)` for determinism
3. **Build Merkle trees:**
   - For each non-empty agent group, call `build_merkle_tree([c.hash for c in chunks], algorithm)`
   - Collect `(agent_type, tree.root)` tuples (skips agents with 0 chunks — empty root `""` is invalid)
4. **Build forest tree:** Call `build_forest_tree(agent_data, algorithm)` to create a two-level proof structure
5. **Flatten:** In sorted-agent order, concatenate all chunks
6. **Reassign indices:** Use `dataclasses.replace(c, chunk_index=i)` to assign 0, 1, 2, ... globally
7. **Return:** `{"forest_tree": forest_tree, "flattened_chunks": flattened}`

**Example:**

Given:

- pubmed agent: chunks with indices [0, 1] (from 2 different sub-queries)
- arxiv agent: chunks with indices [0] (from 1 sub-query)

After corpus assembly:

- Global flattened_chunks: pubmed[0], pubmed[1], arxiv[0] → global indices 0, 1, 2

**Logging:**

- DEBUG: `"corpus_assembler: {N} agent(s), {M} total chunk(s), forest_root={root[:8]}"`

---

### Chunk Selector Node

**File:** `mara/agent/nodes/chunk_selector.py`

**Purpose:** Deduplicate and rank the assembled chunk corpus before passing it to the report synthesizer, preventing LLM context overflow.

**Signature:**

```python
def chunk_selector_node(state: GraphState, config: RunnableConfig) -> dict:
    """Deduplicate, rank, and cap the chunk corpus.

    Reads flattened_chunks from state. Deduplicates by hash, ranks using
    BM25Plus against the original_query, caps at chunk_selector_cap, and
    writes selected_chunks.
    """
```

**Algorithm:**

1. **Extract:** `original_query` and `flattened_chunks` from state
2. **Deduplicate:** Remove chunks with duplicate hashes (keep first occurrence)
3. **Rank:** Score deduplicated chunks with BM25Plus against `original_query` (via `score_chunks_bm25` in `mara/agent/scoring.py`)
4. **Cap:** Take the top `research_config.chunk_selector_cap` (default 50) chunks by score
5. **Return:** `{"selected_chunks": selected}`

**Why BM25Plus?**

BM25Plus uses `log((N+1)/df)` for IDF, which is always non-negative. This avoids the zero/negative scores that `BM25Okapi` produces for small corpora (e.g., a single chunk, or terms appearing in exactly half the corpus).

**Configuration:**

- `chunk_selector_cap` (int, default 50) — maximum chunks passed to the synthesizer; set via `ResearchConfig(chunk_selector_cap=N)`

**Logging:**

- DEBUG: `"chunk_selector: {N_in} → {N_dedup} after dedup → {N_out} selected (cap={cap})"`

---

### Report Synthesizer Node

**File:** `mara/agent/nodes/report_synthesizer.py`

**Purpose:** Call the LLM to synthesize a narrative report from the assembled chunk corpus.

**Signature:**

```python
async def report_synthesizer_node(state: GraphState, config: RunnableConfig) -> dict:
    """Call the LLM to synthesise a report from the assembled chunk corpus.

    Formats each VerifiedChunk as [index:short_hash] text and passes the full
    context to the LLM together with the original query. The raw LLM response
    string is stored in GraphState.report.
    """
```

**Algorithm:**

1. **Extract:** `original_query` and chunks from state — prefers `selected_chunks` (set by chunk_selector), falls back to `flattened_chunks` when `selected_chunks` is absent
2. **Format context:** For each chunk, produce `[{index}:{short_hash}] {text}`
3. **Join context:** All formatted chunks separated by `\n\n`
4. **Prepare user message:** `"Research query: {original_query}\n\nSource excerpts:\n{context}"`
5. **Call LLM:** With system prompt + user message
6. **Extract response:** Get `response.content` (the raw report text)
7. **Return:** `{"report": report}`

**System Prompt:**

```text
You are a research report writer. Given a research query and a set of source excerpts, write a comprehensive, well-structured research report.

Structure: title, executive summary (2-3 sentences), thematic sections with headers, conclusion.
Length: 800-1500 words depending on evidence available.
Citations: cite every claim inline using [ML:index:hash] immediately after the sentence making the claim, where index is the chunk's numeric index and hash is the 8-character short hash from the source list.
Conflicting sources: acknowledge disagreement explicitly rather than picking one side silently.
Thin evidence: if fewer than 3 chunks support a claim, flag it as "limited evidence".
Do not invent facts not present in the source excerpts.
Use clear academic prose.
```

**Citation Format Hint:**

The LLM is asked to use `[ML:index:hash]`, but the downstream `certified_output_node` also accepts `[index:hash]`, `[index]`, and `[N, M, ...]` formats for flexibility.

**Logging:**

- DEBUG: `"report_synthesizer: generated {N}-char report for {query[:60]}"`

---

### Certified Output Node

**File:** `mara/agent/nodes/certified_output.py`

**Purpose:** Parse citations from the report and assemble the final `CertifiedReport`.

**Signature:**

```python
def certified_output_node(state: GraphState, config: RunnableConfig) -> dict:
    """Package the report and provenance into a CertifiedReport.

    Reads original_query, report, forest_tree, and flattened_chunks from state.
    Appends a References section mapping every cited index to its source URL,
    then constructs the immutable CertifiedReport.
    """
```

**Algorithm:**

1. **Extract state:** `original_query`, `report`, `forest_tree`, `flattened_chunks`
2. **Build references:** Parse citations from report and create a References section
3. **Append references:** If any citations found, append to report with `\n## References\n...`
4. **Construct CertifiedReport:** `CertifiedReport(original_query, report, forest_tree, chunks)`
5. **Return:** `{"certified_report": certified}`

#### Citation Parsing Logic (`_build_references_section()`)

The LLM may produce citations in four different formats. The parser handles all of them:

**Format 1: `[ML:N:hash]` (MARA native)**

```regex
\[(?:ML:)?(\d+):[0-9a-f]+\]
```

Captures the numeric index `N` before the hash.

**Format 2: `[N:hash]` (index+hash without prefix)**
Same regex as Format 1 (the `(?:ML:)?` makes the `ML:` optional).

**Format 3: `[N]` (single index shorthand)**

```regex
\[(\d+(?:,\s*\d+)*)\]
```

Captures a single index or comma-separated indices.

**Format 4: `[N, M, ...]` (multi-index)**
Same regex as Format 3; split by comma and parse each part.

**Reference Section Construction:**

1. Build a dict: `chunk_by_index = {c.chunk_index: c for c in chunks}`
2. Extract all cited indices from report using the four regex patterns
3. Filter to valid indices (must exist in chunks): `valid = sorted(i for i in cited if i in chunk_by_index)`
4. If `valid` is empty, return empty string (no References section)
5. Else, construct:

   ```text

   ## References

   [0] https://example.com/paper.pdf
   [1] https://example.com/article.html
   ...
   ```

**Example:**

Report:

```text
According to recent research [ML:0:abc12345], the field is advancing rapidly.
Further evidence [1:def67890] suggests continued growth [3].
```

Parsed indices: `{0, 1, 3}`
References section:

```text

## References

[0] https://arxiv.org/pdf/2024.12345.pdf
[1] https://pubmed.ncbi.nlm.nih.gov/12345678
[3] https://scholar.google.com/scholar?q=...
```

**Logging:**

- DEBUG: `"certified_output: {N} chunk(s), forest_root={root[:8]}"`

---

## Configuration Integration

### `ResearchConfig` (Pydantic BaseSettings)

**File:** `mara/config.py`

The pipeline threads configuration through all nodes via LangGraph's `configurable` dict:

```python
# In run_research():
await graph.ainvoke(
    {"original_query": query},
    config={"configurable": {"research_config": config}},
)

# In each node:
research_config = config["configurable"]["research_config"]
```

**Key fields relevant to the pipeline:**

| Field                | Type                      | Default                              | Used By                                  |
| -------------------- | ------------------------- | ------------------------------------ | ---------------------------------------- |
| `default_model`      | str                       | `"Qwen/Qwen3-30B-A3B-Instruct-2507"` | query_planner, report_synthesizer        |
| `model_overrides`    | dict[str, str]            | `{}`                                 | agents (via `agent.model()`)             |
| `llm_provider`       | str                       | `"featherless-ai"`                   | query_planner, report_synthesizer        |
| `llm_temperature`    | float                     | `0.7`                                | query_planner, report_synthesizer        |
| `llm_top_p`          | float                     | `0.8`                                | query_planner, report_synthesizer        |
| `llm_top_k`          | int                       | `20`                                 | query_planner, report_synthesizer        |
| `llm_max_tokens`     | int                       | `16384`                              | query_planner, report_synthesizer        |
| `hash_algorithm`     | Literal["sha256"]         | `"sha256"`                           | corpus_assembler                         |
| `chunk_selector_cap` | int                       | `50`                                 | chunk_selector                           |
| `chunk_filter`       | ChunkFilter (runtime Any) | `CapFilter()`                        | agents (via `agent._filter()`)           |
| `search_cache`       | SearchCache (runtime Any) | `NoOpCache()`                        | agents (via `agent._fetch_with_retry()`) |

**Lazy Defaults:**

`chunk_filter` and `search_cache` are typed `Any` at runtime (to avoid circular imports) but default to concrete instances via `@model_validator(mode="after")`.

---

## Error Handling

### Node-Level Resilience

**run_agent_node isolation:** If any single agent fails, it logs a WARNING and returns `{"findings": []}`. The exception is not re-raised, so other agents continue executing in parallel.

**query_planner_node fallback:** If LLM response parsing fails, the node falls back to `[SubQuery(original_query)]` and logs a WARNING.

**corpus_assembler_node empty-root skip:** Agents with 0 chunks are skipped when building the ForestTree (empty root `""` is not a valid leaf hash).

**certified_output_node reference parsing:** If no valid citations are found in the report, the References section is omitted entirely (empty string returned by `_build_references_section()`).

### Verification at Construction Time

The pipeline produces `AgentFindings` and `CertifiedReport` objects, both of which perform cryptographic verification during `__post_init__`:

- **`AgentFindings.__post_init__`:** Recomputes the Merkle root from chunks and raises `ValueError("merkle_root mismatch")` if invalid
- **`CertifiedReport`:** A frozen dataclass wrapping verified sub-trees; successful construction = verified

There is no separate external verification step; the immutability and integrity checks are built into the data types themselves.

### Logging Strategy

All nodes use the standard Python `logging` module with a JSON formatter (configured in `mara/logging.py`). Key log calls:

| Level   | Message                                                             | Node               | When               |
| ------- | ------------------------------------------------------------------- | ------------------ | ------------------ |
| DEBUG   | `"query_planner produced N sub-queries..."`                         | query_planner      | After parsing      |
| WARNING | `"query_planner parse failed: ..."`                                 | query_planner      | On parse error     |
| DEBUG   | `"run_agent: agent=X query=..."`                                    | run_agent          | Entry              |
| WARNING | `"run_agent: agent=X failed for query=...: ..."`                    | run_agent          | On agent exception |
| DEBUG   | `"run_agent: agent=X returned N chunk(s)"`                          | run_agent          | Exit               |
| DEBUG   | `"corpus_assembler: N agent(s), M total chunk(s), forest_root=..."` | corpus_assembler   | After assembly     |
| DEBUG   | `"report_synthesizer: generated K-char report for..."`              | report_synthesizer | After LLM call     |
| DEBUG   | `"certified_output: N chunk(s), forest_root=..."`                   | certified_output   | After packaging    |

---

## Usage Examples

### Basic Usage

```python
from mara.agent.graph import run_research
from mara.config import ResearchConfig

# Create a fully-populated config (all API keys from env vars)
config = ResearchConfig()

# Run the pipeline
report = await run_research("What are recent advances in LLM reasoning?", config)

# report is a CertifiedReport
print(report.report)        # The narrative report with citations
print(report.forest_tree)   # The Merkle forest for verification
print(report.chunks)        # All VerifiedChunk objects
```

### Custom Configuration

```python
from mara.agents.cache import InMemoryCache
from mara.agents.filtering import CapFilter

config = ResearchConfig(
    search_cache=InMemoryCache(),           # Cache within process
    chunk_filter=CapFilter(max_chunks_per_agent=100),
    llm_temperature=0.5,                    # More deterministic
    log_level="DEBUG",
)

report = await run_research(query, config)
```

### Accessing Pipeline State Intermediate Results

If you need to inspect intermediate states (e.g., for debugging), you can call `build_graph()` and `ainvoke()` directly:

```python
from mara.agent.graph import build_graph

graph = build_graph()
result = await graph.ainvoke(
    {"original_query": query},
    config={"configurable": {"research_config": config}},
)

# result dict contains all GraphState fields:
print(result["sub_queries"])      # The decomposed sub-queries
print(result["findings"])         # List of AgentFindings
print(result["flattened_chunks"]) # All verified chunks
print(result["report"])           # The synthesized report
print(result["certified_report"]) # The final CertifiedReport
```

### Verification Example

```python
from mara.merkle.proof import verify_proof

report = await run_research(query, config)

# Verify a specific chunk using the forest tree
chunk = report.chunks[0]
proof = report.forest_tree.generate_proof(chunk.hash)  # Two-level proof
is_valid = verify_proof(chunk.hash, proof, report.forest_tree.root)
assert is_valid, "Chunk verification failed!"
```

---

## Build Order and Import Dependencies

To ensure correct initialization, modules must be imported in the following order:

### 1. Merkle Layer

```python
from mara.merkle.hasher import hash_chunk, canonical_serialise
from mara.merkle.tree import build_merkle_tree, MerkleTree
from mara.merkle.proof import verify_proof, generate_proof
from mara.merkle.forest import build_forest_tree, ForestTree
```

**Rationale:** Standalone sub-package; no internal imports; provides the cryptographic foundation.

### 2. Data Types & Configuration

```python
from mara.agents.types import (
    SubQuery, RawChunk, VerifiedChunk, AgentFindings, CertifiedReport
)
from mara.agents.filtering import ChunkFilter, CapFilter, EmbeddingFilter
from mara.config import ResearchConfig
```

**Rationale:** Defines core types and config; no circular dependencies.

### 3. Agent Base & Registry

```python
from mara.agents.base import SpecialistAgent
from mara.agents.registry import _REGISTRY, @agent, get_agents, get_registry_summary
from mara.agents.cache import SearchCache, NoOpCache, InMemoryCache
```

**Rationale:** Base class and registry infrastructure.

### 4. Agent Implementations

```python
import mara.agents.arxiv
import mara.agents.semantic_scholar
import mara.agents.pubmed
import mara.agents.core
import mara.agents.web
```

**Rationale:** Populate `_REGISTRY` via `@agent()` decorators.

### 5. Pipeline State & Nodes

```python
from mara.agent.state import GraphState, AgentRunState
from mara.agent.nodes.query_planner import query_planner_node
from mara.agent.nodes.run_agent import run_agent_node
from mara.agent.nodes.corpus_assembler import corpus_assembler_node
from mara.agent.nodes.chunk_selector import chunk_selector_node
from mara.agent.nodes.report_synthesizer import report_synthesizer_node
from mara.agent.nodes.certified_output import certified_output_node
from mara.agent.edges.routing import route_to_agents
```

**Rationale:** All node dependencies are satisfied. `mara/agent/scoring.py` is a dependency of `chunk_selector.py`.

### 6. Graph

```python
from mara.agent.graph import build_graph, run_research
```

**Rationale:** Can now import and call `build_graph()`.

### Why This Order?

- **No circular imports:** Each layer depends only on earlier layers
- **Registry populated before graph build:** Agent modules are imported before `build_graph()` is called in production
- **Type safety:** All types are defined before being used in graph nodes

### CLI Example (Correct Order)

The CLI respects this order:

```python
# mara/cli.py (simplified)

from mara.logging import configure_logging
from mara.config import ResearchConfig

# Must import agents BEFORE calling run_research()
import mara.agents.arxiv
import mara.agents.semantic_scholar
import mara.agents.pubmed
import mara.agents.core
import mara.agents.web

from mara.agent.graph import run_research

async def run(query: str, config: ResearchConfig) -> CertifiedReport:
    return await run_research(query, config)
```

---

## Summary

The `mara/agent/` package orchestrates a multi-stage research pipeline:

1. **Query planning** decomposes the user's question into focused sub-queries
2. **Parallel routing** sends sub-queries to appropriate agents (directed) or all agents (broadcast fallback)
3. **Parallel execution** runs agents in parallel with error isolation
4. **Corpus assembly** merges findings into a unified Merkle forest
5. **Chunk selection** deduplicates, BM25Plus-ranks, and caps the corpus before synthesis
6. **Report synthesis** calls the LLM to create a narrative with inline citations
7. **Output certification** parses citations and produces a `CertifiedReport` with full provenance

All state flows through a single `GraphState` dict. All nodes are async-compatible and LangGraph-integrated. Configuration is threaded through `RunnableConfig`. Errors are logged but do not crash the pipeline. Cryptographic verification happens at construction time, not after.
