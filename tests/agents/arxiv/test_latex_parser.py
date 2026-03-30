"""Unit tests for mara.agents.arxiv.latex_parser."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mara.agents.arxiv.latex_parser import (
    _pandoc_latex_to_text,
    clean_latex_text,
    extract_sections,
)


class TestExtractSections:
    def test_single_section(self):
        tex = r"\section{Introduction}" + "\nThis is the intro."
        result = extract_sections(tex)
        assert len(result) == 1
        title, body = result[0]
        assert title == "Introduction"
        assert "This is the intro." in body

    def test_multiple_sections(self):
        tex = (
            r"\section{Introduction}" + "\nIntro text.\n"
            r"\section{Methods}" + "\nMethod text.\n"
            r"\section{Results}" + "\nResult text."
        )
        result = extract_sections(tex)
        assert len(result) == 3
        titles = [t for t, _ in result]
        assert titles == ["Introduction", "Methods", "Results"]

    def test_subsection_and_subsubsection(self):
        tex = (
            r"\section{Overview}" + "\nTop level.\n"
            r"\subsection{Detail}" + "\nSub text.\n"
            r"\subsubsection{Deep}" + "\nDeep text."
        )
        result = extract_sections(tex)
        assert len(result) == 3
        assert result[0][0] == "Overview"
        assert result[1][0] == "Detail"
        assert result[2][0] == "Deep"

    def test_starred_variants(self):
        tex = r"\section*{Unnumbered}" + "\nBody text."
        result = extract_sections(tex)
        assert result[0][0] == "Unnumbered"
        assert "Body text." in result[0][1]

    def test_no_section_commands_returns_full_text(self):
        tex = "Just a paragraph with no sections."
        result = extract_sections(tex)
        assert result == [("", tex)]

    def test_empty_section_body_skipped(self):
        # Only whitespace between sections → both bodies are empty after strip
        tex = r"\section{A}" + "   \n" + r"\section{B}" + "\nReal content."
        result = extract_sections(tex)
        # Section A has only whitespace body → skipped; Section B has content
        assert len(result) == 1
        assert result[0][0] == "B"

    def test_all_sections_empty_falls_back_to_full_text(self):
        # All sections empty → falls back to full text
        tex = r"\section{A}" + "   " + r"\section{B}" + "   "
        result = extract_sections(tex)
        assert result == [("", tex)]

    def test_text_before_first_section_ignored(self):
        # Preamble text before the first \section is not included in any section
        tex = "Preamble.\n" + r"\section{Intro}" + "\nIntro text."
        result = extract_sections(tex)
        assert len(result) == 1
        assert result[0][0] == "Intro"

    def test_exception_falls_back_to_full_text(self, monkeypatch):
        # Force an exception inside the function
        import mara.agents.arxiv.latex_parser as parser

        monkeypatch.setattr(parser, "_SECTION_RE", None)  # causes AttributeError
        result = extract_sections("some latex")
        assert result == [("", "some latex")]


class TestPandocLatexToText:
    def test_converts_plain_text(self):
        result = _pandoc_latex_to_text("Hello world.")
        assert result is not None
        assert "Hello" in result
        assert "world" in result

    def test_handles_href_with_nested_command(self):
        # This is the exact pattern that crashes pylatexenc
        result = _pandoc_latex_to_text(
            r"\href{https://example.com}{\textbf{Click here}}"
        )
        assert result is not None
        assert "Click here" in result

    def test_handles_includegraphics(self):
        result = _pandoc_latex_to_text(r"\includegraphics[width=0.5\linewidth]{fig1}")
        assert result is not None  # should not crash

    def test_handles_math(self):
        result = _pandoc_latex_to_text(r"Equation $x^2 + y^2 = z^2$ holds.")
        assert result is not None
        assert result  # non-empty

    def test_returns_none_when_pandoc_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _pandoc_latex_to_text("Hello world.")
        assert result is None

    def test_returns_none_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pandoc", 5)):
            result = _pandoc_latex_to_text("Hello world.")
        assert result is None

    def test_returns_none_on_nonzero_exit(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = b"pandoc error"
        with patch("subprocess.run", return_value=mock_proc):
            result = _pandoc_latex_to_text("Hello world.")
        assert result is None

    def test_empty_input(self):
        result = _pandoc_latex_to_text("")
        # pandoc on empty input returns empty string → stripped to "" → None not expected
        # empty string is falsy but not None; caller checks `is not None`
        assert result is not None
        assert result == ""


class TestCleanLatexText:
    def test_plain_text_unchanged(self):
        result = clean_latex_text("Hello world.")
        assert "Hello" in result
        assert "world" in result

    def test_empty_input(self):
        assert clean_latex_text("") == ""

    def test_removes_commands(self):
        result = clean_latex_text(r"\textbf{important}")
        assert "important" in result

    def test_href_with_nested_command(self):
        # This crashed pylatexenc; pandoc should handle it
        result = clean_latex_text(r"\href{https://example.com}{\textbf{Click here}}")
        assert "Click here" in result

    def test_strips_latex_comments(self):
        result = clean_latex_text("Hello % this is a comment\nworld.")
        assert "comment" not in result

    def test_math_section_handled(self):
        result = clean_latex_text(r"See equation $x^2 + y^2 = z^2$ for proof.")
        assert result  # non-empty

    def test_pylatexenc_fallback_when_pandoc_unavailable(self):
        # When pandoc is not found, pylatexenc handles simple input
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = clean_latex_text(r"\textbf{important}")
        assert "important" in result

    def test_regex_fallback_when_both_pandoc_and_pylatexenc_fail(self, monkeypatch):
        # Make pandoc unavailable AND pylatexenc fail → regex fallback
        def fail_import(name, *args, **kwargs):
            if name == "pylatexenc.latex2text":
                raise ImportError("forced failure")
            import builtins
            return builtins.__import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fail_import)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = clean_latex_text(r"% comment" + "\n" + r"\textbf{word}")
        assert "word" in result
        assert "comment" not in result
