"""Report synthesizer node — generates the narrative report from assembled chunks."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from mara.agent.state import GraphState
from mara.llm import make_llm

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research report writer. "
    "Given a research query and a set of source excerpts, write a comprehensive, "
    "well-structured research report that synthesises the key findings. "
    "Structure your report as: title, executive summary (2-3 sentences), "
    "thematic sections with headers, and a conclusion. "
    "Target 800-1500 words depending on the evidence available. "
    "Cite every claim inline using the format [ML:index:hash] where index is the "
    "chunk's numeric index and hash is the 8-character short hash shown in the "
    "source list. Place citations immediately after the sentence making the claim. "
    "When sources conflict, acknowledge the disagreement explicitly rather than "
    "picking one side silently. "
    "When fewer than 3 source excerpts support a claim, flag it as "
    "'limited evidence suggests'. "
    "Do not invent facts beyond what the source excerpts support. "
    "Use clear academic prose."
)


async def report_synthesizer_node(state: GraphState, config: RunnableConfig) -> dict:
    """Call the LLM to synthesise a report from the assembled chunk corpus.

    Formats each ``VerifiedChunk`` as ``[index:short_hash] text`` and passes
    the full context to the LLM together with the original query.  The raw
    LLM response string is stored in ``GraphState.report``.
    """
    research_config = config["configurable"]["research_config"]
    original_query: str = state["original_query"]
    flattened_chunks = state.get("selected_chunks") or state.get("flattened_chunks", [])

    context = "\n\n".join(
        f"[{c.chunk_index}:{c.short_hash}] {c.text}" for c in flattened_chunks
    )

    llm = make_llm(
        model=research_config.default_model,
        hf_token=research_config.hf_token,
        max_new_tokens=research_config.llm_max_tokens,
        provider=research_config.llm_provider,
        temperature=research_config.synthesizer_temperature,
        top_p=research_config.llm_top_p,
        top_k=research_config.llm_top_k,
    )

    user_content = f"Research query: {original_query}\n\nSource excerpts:\n{context}"
    response = await llm.ainvoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
    )

    report: str = response.content
    _log.debug(
        "report_synthesizer: generated %d-char report for %r",
        len(report),
        original_query[:60],
    )
    return {"report": report}
