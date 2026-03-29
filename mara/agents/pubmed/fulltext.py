"""PubMed / PMC full-text XML parsers.

Pure functions — no HTTP calls; those live in agent.py.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


def parse_pmc_sections(xml_text: str) -> list[str]:
    """Extract section texts from a PMC full-text XML response.

    Each ``<sec>`` element (including nested ones) is returned as a separate
    string containing the section title (if present) followed by its direct
    child ``<p>`` paragraph texts.  Sections that yield no non-whitespace
    paragraph text are skipped.

    Args:
        xml_text: Raw XML string from an efetch PMC full-text call.

    Returns:
        List of section text strings, one per ``<sec>`` with usable content.
        Returns an empty list on parse errors or if no sections are found.
    """
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    sections: list[str] = []
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title = (
            title_el.text.strip()
            if title_el is not None and title_el.text
            else ""
        )

        paragraphs: list[str] = []
        for p in sec.findall("p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        if not paragraphs:
            continue

        parts = []
        if title:
            parts.append(title)
        parts.extend(paragraphs)
        sections.append("\n\n".join(parts))

    return sections


def parse_abstract_xml(xml_text: str) -> str:
    """Extract abstract text from a PubMed efetch XML response.

    Concatenates all ``<AbstractText>`` element contents (space-separated),
    which handles structured abstracts with labelled sections.

    Args:
        xml_text: Raw XML string from an efetch PubMed abstract call.

    Returns:
        Stripped abstract string, or ``""`` if none found or on parse error.
    """
    if not xml_text:
        return ""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    parts: list[str] = []
    for el in root.iter("AbstractText"):
        text = "".join(el.itertext()).strip()
        if text:
            parts.append(text)

    return " ".join(parts).strip()
