"""Tests for mara.agents.pubmed.search."""

from __future__ import annotations

import pytest

from mara.agents.pubmed.search import parse_article_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_esummary_item(
    uid: str = "12345",
    title: str = "Test Paper",
    article_ids: list[dict] | None = None,
) -> dict:
    """Build a minimal esummary result item."""
    if article_ids is None:
        article_ids = [{"idtype": "pmc", "value": "PMC123456"}]
    return {"uid": uid, "title": title, "articleids": article_ids}


# ---------------------------------------------------------------------------
# parse_article_metadata
# ---------------------------------------------------------------------------


class TestParseArticleMetadata:
    def test_happy_path_with_pmc_id(self):
        item = _make_esummary_item()
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmid"] == "12345"
        assert result["title"] == "Test Paper"
        assert result["pmc_id"] == "PMC123456"

    def test_happy_path_without_pmc_id(self):
        item = _make_esummary_item(article_ids=[{"idtype": "doi", "value": "10.1234/x"}])
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmid"] == "12345"
        assert result["pmc_id"] is None

    def test_missing_uid_returns_none(self):
        item = {"title": "Some Title", "articleids": []}
        assert parse_article_metadata(item) is None

    def test_empty_uid_returns_none(self):
        item = _make_esummary_item(uid="   ")
        assert parse_article_metadata(item) is None

    def test_missing_title_returns_none(self):
        item = {"uid": "12345", "articleids": []}
        assert parse_article_metadata(item) is None

    def test_empty_title_returns_none(self):
        item = _make_esummary_item(title="   ")
        assert parse_article_metadata(item) is None

    def test_missing_articleids_pmc_id_is_none(self):
        item = {"uid": "99999", "title": "Paper Without IDs"}
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmc_id"] is None

    def test_empty_articleids_pmc_id_is_none(self):
        item = _make_esummary_item(article_ids=[])
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmc_id"] is None

    def test_pmc_value_whitespace_only_treated_as_absent(self):
        item = _make_esummary_item(article_ids=[{"idtype": "pmc", "value": "   "}])
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmc_id"] is None

    def test_multiple_articleids_picks_pmc(self):
        item = _make_esummary_item(
            article_ids=[
                {"idtype": "doi", "value": "10.1234/x"},
                {"idtype": "pmc", "value": "PMC999"},
                {"idtype": "pubmed", "value": "12345"},
            ]
        )
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmc_id"] == "PMC999"

    def test_title_is_stripped(self):
        item = _make_esummary_item(title="  Trimmed Title  ", article_ids=[])
        result = parse_article_metadata(item)
        assert result is not None
        assert result["title"] == "Trimmed Title"

    def test_uid_is_stripped(self):
        item = _make_esummary_item(uid="  12345  ", article_ids=[])
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmid"] == "12345"

    @pytest.mark.parametrize(
        "uid,title",
        [
            ("111", "Paper A"),
            ("222", "Paper B"),
        ],
        ids=["paper-a", "paper-b"],
    )
    def test_parametrized_uid_and_title(self, uid, title):
        item = _make_esummary_item(uid=uid, title=title, article_ids=[])
        result = parse_article_metadata(item)
        assert result is not None
        assert result["pmid"] == uid
        assert result["title"] == title
