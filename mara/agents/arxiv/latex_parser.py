"""LaTeX source parser for the ArXiv agent.

Provides section extraction and text cleaning.  Both functions are
designed to be fault-tolerant: any parse error falls back to returning
the full input text rather than raising.
"""

from __future__ import annotations

import logging
import re

_log = logging.getLogger(__name__)

# Matches \section, \subsection, \subsubsection (with optional star variant).
# Captures the title inside the braces.  Does not handle nested braces in
# titles (e.g. \section{A {nested} title}) — edge case not worth the
# complexity for our use-case.
_SECTION_RE = re.compile(
    r"\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}",
    re.MULTILINE,
)


def extract_sections(latex_text: str) -> list[tuple[str, str]]:
    """Split *latex_text* into ``(title, body)`` pairs on section commands.

    Recognises ``\\section``, ``\\subsection``, and ``\\subsubsection``
    (starred variants included).

    Returns:
        A list of ``(title, body)`` tuples.  ``title`` is the text inside
        the section braces; ``body`` is the text between this command and
        the next section command.  Sections whose body is empty after
        stripping are omitted.

        If no section commands are found, or if every section has an empty
        body, the function returns a single tuple ``("", latex_text)``.

        On any exception, falls back to the same single-tuple behaviour.
    """
    try:
        matches = list(_SECTION_RE.finditer(latex_text))
        if not matches:
            return [("", latex_text)]

        sections: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            title = match.group(1)
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(latex_text)
            body = latex_text[body_start:body_end].strip()
            if body:
                sections.append((title, body))

        return sections if sections else [("", latex_text)]

    except Exception:
        _log.debug("extract_sections failed, returning full text as single chunk")
        return [("", latex_text)]


def clean_latex_text(text: str) -> str:
    """Convert LaTeX markup to plain text.

    Uses ``pylatexenc.latex2text.LatexNodes2Text`` for the primary
    conversion.  On any failure, falls back to a simple regex-based
    stripping of comments and common commands.
    """
    if not text:
        return ""

    try:
        from pylatexenc.latex2text import LatexNodes2Text  # noqa: PLC0415

        return LatexNodes2Text().latex_to_text(text)
    except Exception:
        _log.debug("pylatexenc conversion failed, using regex fallback")
        # Strip LaTeX comments
        cleaned = re.sub(r"%[^\n]*", "", text)
        # Replace \cmd{content} → content (preserve content)
        cleaned = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", cleaned)
        # Strip remaining bare commands
        cleaned = re.sub(r"\\[a-zA-Z]+\*?", " ", cleaned)
        return cleaned.strip()
