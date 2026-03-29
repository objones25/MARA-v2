"""URL tier filter and optional LLM ranking for the web agent.

Two-pass filter:
1. Domain-tier regex (BLOCK / DEFAULT / GOOD / PRIORITY) — synchronous.
2. Optional LLM relevance ranking with hallucination guard — async.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from enum import IntEnum

from langchain_core.messages import HumanMessage

from mara.config import ResearchConfig
from mara.llm import make_llm

_log = logging.getLogger(__name__)


class DomainTier(IntEnum):
    """Relative quality tier for a URL's domain.

    Higher values are preferred; BLOCK URLs are excluded entirely.
    """

    BLOCK = 0
    DEFAULT = 1
    GOOD = 2
    PRIORITY = 3


# ---------------------------------------------------------------------------
# Hostname-level regex patterns — evaluated in BLOCK → PRIORITY → GOOD order;
# first match wins.  Unmatched hostnames fall through to DEFAULT.
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"(?:^|\.)(?:"
    r"facebook|twitter|x|instagram|tiktok|pinterest|reddit|quora"
    r"|linkedin|youtube|tumblr|snapchat|discord|vk|weibo"
    r")\.com$",
    re.IGNORECASE,
)

_PRIORITY_RE = re.compile(
    r"(?:"
    r"\.edu$"
    r"|\.gov$"
    r"|\.ac\.(?:uk|jp|au|de|fr|in|cn|kr|nl|nz|za|be|it|se|es|dk|fi|no|pl|at|ch|pt|ru)$"
    r"|(?:^|\.)(?:"
    r"nature|sciencedirect|springer|wiley|tandfonline|jstor"
    r"|researchgate|academia|ieee|acm"
    r")\.(?:org|com|gov)$"
    r")",
    re.IGNORECASE,
)

_GOOD_RE = re.compile(
    r"(?:^|\.)(?:"
    r"wikipedia|bbc|reuters|apnews|nytimes|theguardian|economist"
    r"|ft|wsj|bloomberg|pbs|techcrunch|wired|arstechnica|theatlantic"
    r"|newyorker|vox|slate|salon"
    r")\.(?:org|com)$",
    re.IGNORECASE,
)


def _classify_hostname(hostname: str) -> DomainTier:
    if _BLOCK_RE.search(hostname):
        return DomainTier.BLOCK
    if _PRIORITY_RE.search(hostname):
        return DomainTier.PRIORITY
    if _GOOD_RE.search(hostname):
        return DomainTier.GOOD
    return DomainTier.DEFAULT


def classify_url(url: str) -> DomainTier:
    """Classify *url* into a ``DomainTier`` based on its hostname."""
    try:
        hostname = urllib.parse.urlparse(url).hostname or ""
    except Exception:  # pragma: no cover
        return DomainTier.DEFAULT
    if not hostname:
        return DomainTier.DEFAULT
    return _classify_hostname(hostname)


def filter_urls_by_tier(urls: list[str]) -> list[str]:
    """Remove BLOCK-tier URLs and sort survivors by tier descending.

    URLs within the same tier retain their original relative order.
    """
    classified = [(url, classify_url(url)) for url in urls]
    classified = [(url, tier) for url, tier in classified if tier != DomainTier.BLOCK]
    classified.sort(key=lambda x: x[1], reverse=True)
    return [url for url, _ in classified]


async def rank_urls_with_llm(
    urls: list[str], query: str, config: ResearchConfig
) -> list[str]:
    """Use an LLM to rank and filter *urls* by relevance to *query*.

    Returns a subset of *urls* (preserving the LLM's preferred order).
    Any URL the LLM fabricates that is not in the original set is silently
    discarded (hallucination guard).  Falls back to the unranked *urls* list
    when the LLM call fails or produces no valid URLs.
    """
    url_set = set(urls)
    numbered = "\n".join(f"{i + 1}. {url}" for i, url in enumerate(urls))
    prompt = (
        f"You are a URL relevance filter for a research assistant.\n"
        f"Research query: {query}\n\n"
        f"From the following URLs, return ONLY the URLs that are likely to "
        f"contain information relevant to the research query. "
        f"Return one URL per line, exactly as given, no numbering, no extra text.\n\n"
        f"{numbered}"
    )

    try:
        llm = make_llm(
            model=config.default_model,
            hf_token=config.hf_token,
            max_new_tokens=config.llm_max_tokens,
            provider=config.llm_provider,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            top_k=config.llm_top_k,
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        response_text: str = response.content if hasattr(response, "content") else str(response)

        result: list[str] = []
        for line in response_text.splitlines():
            line = line.strip()
            if line in url_set:
                result.append(line)

        if result:
            return result

        _log.debug("LLM URL ranking returned no valid URLs; using unranked list")
        return urls

    except Exception as exc:
        _log.debug("LLM URL ranking failed: %s; using unranked list", exc)
        return urls
