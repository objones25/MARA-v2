"""HTTP fetching and content extraction helpers for the ArXiv agent."""

from __future__ import annotations

import io
import logging
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

from mara.agents.types import RawChunk
from mara.agents.utils.pdf import extract_pdf_text

_log = logging.getLogger(__name__)

LATEX = "latex"


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
            _log.debug("tarball fetch: HTTP %d for %s", resp.status_code, versioned_id)
            return None
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type.lower():
            _log.debug("tarball endpoint returned PDF for %s", versioned_id)
            return None
        return bytes(resp.content)
    except Exception as exc:
        _log.debug("tarball fetch failed for %s: %s", versioned_id, exc)
        return None


def compile_latex_to_pdf(tarball_bytes: bytes) -> bytes:
    """Extract a LaTeX tarball, compile main.tex with pdflatex, and return PDF bytes.

    Locates the entry point by preferring ``main.tex``; if absent, scans for
    the first ``.tex`` file containing ``\\documentclass``.

    Raises ``RuntimeError`` if pdflatex does not produce a PDF.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            tar.extractall(tmpdir)

        main_tex = tmp / "main.tex"
        if not main_tex.exists():
            for tex_file in sorted(tmp.rglob("*.tex")):
                if r"\documentclass" in tex_file.read_text(errors="replace"):
                    main_tex = tex_file
                    break

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(main_tex)],
            cwd=tmpdir,
            capture_output=True,
            timeout=120,
        )

        pdf_path = main_tex.with_suffix(".pdf")
        if not pdf_path.exists():
            raise RuntimeError(f"pdflatex did not produce {pdf_path.name}")
        return pdf_path.read_bytes()


def chunks_from_pdf(
    pdf_bytes: bytes,
    canonical_url: str,
    sub_query: str,
    retrieved_at: str,
) -> list[RawChunk]:
    """Extract text from *pdf_bytes* and wrap in a single ``RawChunk``."""
    text = extract_pdf_text(pdf_bytes)
    if not text:
        return []
    return [
        RawChunk(
            url=canonical_url,
            text=text,
            retrieved_at=retrieved_at,
            source_type=LATEX,
            sub_query=sub_query,
        )
    ]
