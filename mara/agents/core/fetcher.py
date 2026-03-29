"""PDF text extraction helper for the CORE agent."""

from __future__ import annotations

import io
import logging

_log = logging.getLogger(__name__)


def extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """Extract plain text from *pdf_bytes* using pypdf.

    Returns the concatenated page texts, or ``None`` when extraction
    fails or produces no text.
    """
    try:
        import pypdf  # noqa: PLC0415

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        result = "\n\n".join(pages).strip()
        return result if result else None
    except Exception as exc:
        _log.debug("PDF text extraction failed: %s", exc)
        return None
