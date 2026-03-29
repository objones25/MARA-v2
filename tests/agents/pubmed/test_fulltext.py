"""Tests for mara.agents.pubmed.fulltext."""

from __future__ import annotations

import pytest

from mara.agents.pubmed.fulltext import parse_abstract_xml, parse_pmc_sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_pmc(body: str) -> str:
    return f"<article><body>{body}</body></article>"


def _single_sec(title: str, *paragraphs: str) -> str:
    paras = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<sec><title>{title}</title>{paras}</sec>"


# ---------------------------------------------------------------------------
# parse_pmc_sections
# ---------------------------------------------------------------------------


class TestParsePMCSections:
    def test_happy_path_single_section(self):
        xml = _wrap_pmc(_single_sec("Introduction", "This is the intro."))
        result = parse_pmc_sections(xml)
        assert len(result) == 1
        assert "Introduction" in result[0]
        assert "This is the intro." in result[0]

    def test_multiple_sections(self):
        xml = _wrap_pmc(
            _single_sec("Introduction", "Intro text.")
            + _single_sec("Methods", "Method text.")
        )
        result = parse_pmc_sections(xml)
        assert len(result) == 2

    def test_multiple_paragraphs_in_section(self):
        xml = _wrap_pmc(_single_sec("Results", "Para 1.", "Para 2.", "Para 3."))
        result = parse_pmc_sections(xml)
        assert len(result) == 1
        assert "Para 1." in result[0]
        assert "Para 2." in result[0]
        assert "Para 3." in result[0]

    def test_section_without_title(self):
        xml = _wrap_pmc("<sec><p>No title here.</p></sec>")
        result = parse_pmc_sections(xml)
        assert len(result) == 1
        assert "No title here." in result[0]

    def test_nested_sections_each_returned(self):
        nested = (
            "<sec>"
            "<title>Parent</title>"
            "<p>Parent para.</p>"
            "<sec><title>Child</title><p>Child para.</p></sec>"
            "</sec>"
        )
        xml = _wrap_pmc(nested)
        result = parse_pmc_sections(xml)
        # Both parent and child sec appear; at least one has "Parent para."
        assert any("Parent para." in r for r in result)
        assert any("Child para." in r for r in result)

    def test_whitespace_only_paragraph_skipped(self):
        xml = _wrap_pmc("<sec><title>Empty</title><p>   </p></sec>")
        result = parse_pmc_sections(xml)
        assert result == []

    def test_section_with_only_whitespace_paragraphs_skipped(self):
        xml = _wrap_pmc("<sec><title>Empty</title><p>  </p><p>\n</p></sec>")
        result = parse_pmc_sections(xml)
        assert result == []

    def test_no_sections_returns_empty(self):
        xml = "<article><body><p>No sections here.</p></body></article>"
        result = parse_pmc_sections(xml)
        assert result == []

    def test_malformed_xml_returns_empty(self):
        assert parse_pmc_sections("not xml at all") == []

    def test_empty_string_returns_empty(self):
        assert parse_pmc_sections("") == []

    def test_text_is_stripped(self):
        xml = _wrap_pmc("<sec><title>  Trimmed  </title><p>  Content.  </p></sec>")
        result = parse_pmc_sections(xml)
        assert result[0].startswith("Trimmed")
        assert "Content." in result[0]


# ---------------------------------------------------------------------------
# parse_abstract_xml
# ---------------------------------------------------------------------------


class TestParseAbstractXml:
    def test_happy_path_single_abstract(self):
        xml = (
            "<PubmedArticle><MedlineCitation><Article>"
            "<Abstract><AbstractText>This is the abstract.</AbstractText></Abstract>"
            "</Article></MedlineCitation></PubmedArticle>"
        )
        result = parse_abstract_xml(xml)
        assert result == "This is the abstract."

    def test_multiple_abstract_text_elements_joined(self):
        xml = (
            "<PubmedArticle><Abstract>"
            "<AbstractText Label=\"Background\">Background text.</AbstractText>"
            "<AbstractText Label=\"Methods\">Methods text.</AbstractText>"
            "</Abstract></PubmedArticle>"
        )
        result = parse_abstract_xml(xml)
        assert "Background text." in result
        assert "Methods text." in result

    def test_no_abstract_text_returns_empty(self):
        xml = "<PubmedArticle><MedlineCitation><Article></Article></MedlineCitation></PubmedArticle>"
        assert parse_abstract_xml(xml) == ""

    def test_malformed_xml_returns_empty(self):
        assert parse_abstract_xml("not xml") == ""

    def test_empty_string_returns_empty(self):
        assert parse_abstract_xml("") == ""

    def test_whitespace_only_abstract_returns_empty(self):
        xml = "<PubmedArticle><Abstract><AbstractText>   </AbstractText></Abstract></PubmedArticle>"
        assert parse_abstract_xml(xml) == ""

    def test_abstract_text_is_stripped(self):
        xml = "<PubmedArticle><Abstract><AbstractText>  Trimmed.  </AbstractText></Abstract></PubmedArticle>"
        result = parse_abstract_xml(xml)
        assert result == "Trimmed."
