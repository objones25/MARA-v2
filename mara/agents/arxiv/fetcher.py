"""HTTP fetching and content extraction helpers for the ArXiv agent.

Implements the four-level content fallback chain:

1. Source tarball → ``.tex`` files → LaTeX section chunks (``"latex"``)
2. Source tarball → embedded PDF → pypdf text  (``"pdf_from_tarball"``)
3. Versioned PDF URL → pypdf text              (``"pdf_rendered"``)
4. Abstract text only                          (``"abstract_only"``)

All network calls are the caller's responsibility (``ArxivAgent._search``
passes in an ``httpx.AsyncClient``).  The helpers here are synchronous
and purely transform bytes → ``RawChunk`` lists.
"""

from __future__ import annotations

import io
import logging
import tarfile
from datetime import datetime, timezone

import httpx

from mara.agents.arxiv.latex_parser import clean_latex_text, extract_sections
from mara.agents.types import RawChunk
from mara.agents.utils.pdf import extract_pdf_text

_log = logging.getLogger(__name__)

# source_type constants — referenced by agent.py and tests
LATEX = "latex"
PDF_FROM_TARBALL = "pdf_from_tarball"
PDF_RENDERED = "pdf_rendered"
ABSTRACT_ONLY = "abstract_only"


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


async def fetch_source_tarball(
    client: httpx.AsyncClient, versioned_id: str
) -> bytes | None:
    """Fetch the source tarball for *versioned_id* from ``arxiv.org/src``.

    Returns the raw bytes on success, or ``None`` when:
    - the server responds with a non-200 status,
    - the content-type indicates a PDF (submission was PDF-only), or
    - any network exception occurs.
    """
    url = f"https://arxiv.org/src/{versioned_id}"
    try:
        resp = await client.get(url, follow_redirects=True)
        if resp.status_code != 200:
            _log.debug(
                "tarball fetch: HTTP %d for %s", resp.status_code, versioned_id
            )
            return None
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            # ArXiv returns the compiled PDF directly for PDF-only submissions
            _log.debug("tarball endpoint returned PDF for %s", versioned_id)
            return None
        return bytes(resp.content)
    except Exception as exc:
        _log.debug("tarball fetch failed for %s: %s", versioned_id, exc)
        return None


def extract_from_tarball(
    tarball_bytes: bytes,
) -> tuple[list[str], bytes | None]:
    """Extract LaTeX source text and an optional PDF from a ``.tar.gz`` blob.

    Returns ``(tex_contents, pdf_bytes)`` where:
    - ``tex_contents``: decoded text of every ``.tex`` member, in archive order.
    - ``pdf_bytes``: raw bytes of the first ``.pdf`` member, or ``None``.

    On any extraction error the function logs at DEBUG and returns ``([], None)``.
    """
    tex_contents: list[str] = []
    pdf_bytes: bytes | None = None

    try:
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read()
                name = member.name.lower()
                if name.endswith(".tex"):
                    tex_contents.append(raw.decode("utf-8", errors="replace"))
                elif name.endswith(".pdf") and pdf_bytes is None:
                    pdf_bytes = raw
    except Exception as exc:
        _log.debug("tarball extraction failed: %s", exc)
        return [], None

    return tex_contents, pdf_bytes


def chunks_from_latex(
    tex_contents: list[str],
    canonical_url: str,
    sub_query: str,
    retrieved_at: str,
) -> list[RawChunk]:
    """Convert ``.tex`` source strings into one ``RawChunk`` per section.

    Empty sections (after LaTeX cleaning) are silently skipped.
    ``source_type`` is set to ``LATEX`` so the agent's ``_chunk()``
    override can bypass the sliding window.
    """
    chunks: list[RawChunk] = []
    for tex in tex_contents:
        for title, body in extract_sections(tex):
            cleaned = clean_latex_text(body).strip()
            if not cleaned:
                continue
            text = f"{title}\n\n{cleaned}".strip() if title else cleaned
            chunks.append(
                RawChunk(
                    url=canonical_url,
                    text=text,
                    retrieved_at=retrieved_at,
                    source_type=LATEX,
                    sub_query=sub_query,
                )
            )
    return chunks


def chunks_from_pdf(
    pdf_bytes: bytes,
    canonical_url: str,
    sub_query: str,
    retrieved_at: str,
    source_type: str,
) -> list[RawChunk]:
    """Extract text from *pdf_bytes* and wrap in a single ``RawChunk``.

    Returns an empty list when text extraction fails or yields no text.
    """
    text = extract_pdf_text(pdf_bytes)
    if not text:
        return []
    return [
        RawChunk(
            url=canonical_url,
            text=text,
            retrieved_at=retrieved_at,
            source_type=source_type,
            sub_query=sub_query,
        )
    ]


def chunks_from_abstract(
    abstract: str,
    canonical_url: str,
    sub_query: str,
    retrieved_at: str,
) -> list[RawChunk]:
    """Wrap *abstract* text in a single ``RawChunk`` of type ``ABSTRACT_ONLY``."""
    return [
        RawChunk(
            url=canonical_url,
            text=abstract,
            retrieved_at=retrieved_at,
            source_type=ABSTRACT_ONLY,
            sub_query=sub_query,
        )
    ]
