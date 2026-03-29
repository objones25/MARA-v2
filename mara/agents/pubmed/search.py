"""PubMed esummary response parser.

Pure functions — no HTTP calls; those live in agent.py.
"""

from __future__ import annotations


def parse_article_metadata(item: dict) -> dict | None:
    """Extract ``{pmid, title, pmc_id}`` from one esummary result item.

    Returns ``None`` if the item is missing a required field (uid or title),
    so callers can skip the article without raising.

    Args:
        item: One entry from ``esummary`` ``result`` dict (keyed by PMID).

    Returns:
        Dict with ``pmid`` (str), ``title`` (str), ``pmc_id`` (str | None),
        or ``None`` if the item should be skipped.
    """
    pmid = item.get("uid", "").strip()
    title = item.get("title", "").strip()
    if not pmid or not title:
        return None

    pmc_id: str | None = None
    for aid in item.get("articleids", []):
        if aid.get("idtype") == "pmc":
            val = aid.get("value", "").strip()
            if val:
                pmc_id = val
                break

    return {"pmid": pmid, "title": title, "pmc_id": pmc_id}
