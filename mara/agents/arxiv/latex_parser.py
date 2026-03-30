"""LaTeX source parser for the ArXiv agent.

Provides section extraction and text cleaning.  Both functions are
designed to be fault-tolerant: any parse error falls back to returning
the full input text rather than raising.

Conversion pipeline for ``clean_latex_text``:
1. **pandoc** — subprocess call; handles any valid LaTeX including
   ``\\href``, ``\\includegraphics``, custom environments, and math.
   Falls back when pandoc is not installed or times out.
2. **pylatexenc** — pure-Python; covers simpler documents but crashes
   on ``\\href{url}{\\cmd{...}}`` due to a mismatch between its walker
   and renderer macro-arg counts.
3. **regex** — last-resort strip of comments and common commands.
"""

from __future__ import annotations

import logging
import re
import subprocess

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


def _pandoc_latex_to_text(text: str) -> str | None:
    """Convert *text* via ``pandoc --from=latex --to=plain``.

    Returns the plain-text string on success, or ``None`` when:
    - pandoc is not installed (``FileNotFoundError``),
    - the conversion times out (>5 s), or
    - pandoc exits with a non-zero return code.

    Logs at DEBUG on expected failures (missing binary, timeout);
    logs WARNING for unexpected non-zero exit codes.
    """
    # Pandoc requires a complete LaTeX document.  Section bodies extracted
    # from larger files are fragments — wrap them in minimal boilerplate.
    # Also strip any stray \end{document} that may appear at the tail of
    # the last section returned by extract_sections.
    if r"\begin{document}" not in text:
        body = text.replace(r"\end{document}", "")
        pandoc_input = (
            r"\documentclass{article}\begin{document}" + body + r"\end{document}"
        )
    else:
        pandoc_input = text

    try:
        proc = subprocess.run(
            ["pandoc", "--from=latex", "--to=plain", "--wrap=none"],
            input=pandoc_input.encode("utf-8", errors="replace"),
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        _log.debug("pandoc not found; falling back to pylatexenc")
        return None
    except subprocess.TimeoutExpired:
        _log.debug("pandoc timed out; falling back to pylatexenc")
        return None

    if proc.returncode != 0:
        _log.warning(
            "pandoc exited %d: %s",
            proc.returncode,
            proc.stderr.decode("utf-8", errors="replace")[:200],
        )
        return None

    return proc.stdout.decode("utf-8", errors="replace").strip()


def clean_latex_text(text: str) -> str:
    """Convert LaTeX markup to plain text.

    Tries three strategies in order, returning the first success:

    1. **pandoc** subprocess (best quality; handles ``\\href``, math,
       custom environments).
    2. **pylatexenc** ``LatexNodes2Text`` (pure-Python; fails on some
       ``\\href{url}{\\cmd{...}}`` patterns).
    3. Regex strip of comments and common commands (last resort).
    """
    if not text:
        return ""

    result = _pandoc_latex_to_text(text)
    if result is not None:
        return result

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
