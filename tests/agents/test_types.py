"""Tests for mara/agents/types.py — VerifiedChunk, AgentFindings, RawChunk, SubQuery."""

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from mara.agents.types import AgentFindings, CertifiedReport, RawChunk, SubQuery, VerifiedChunk
from mara.merkle.forest import ForestTree
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2026-03-28T00:00:00+00:00"


def _make_chunk(
    url: str = "https://example.com",
    text: str = "hello",
    retrieved_at: str = _TS,
    source_type: str = "abstract",
    sub_query: str = "foo",
    chunk_index: int = 0,
) -> VerifiedChunk:
    h = hash_chunk(url, text, retrieved_at, "sha256")
    return VerifiedChunk(
        hash=h,
        url=url,
        text=text,
        retrieved_at=retrieved_at,
        source_type=source_type,
        sub_query=sub_query,
        chunk_index=chunk_index,
    )


def _make_findings(chunks: list[VerifiedChunk], agent_type: str = "arxiv", query: str = "q") -> AgentFindings:
    tree = build_merkle_tree([c.hash for c in chunks], "sha256")
    return AgentFindings(
        agent_type=agent_type,
        query=query,
        chunks=tuple(chunks),
        merkle_root=tree.root,
        merkle_tree=tree,
    )


# ---------------------------------------------------------------------------
# SubQuery
# ---------------------------------------------------------------------------


class TestSubQuery:
    def test_has_query_field(self):
        sq = SubQuery(query="what is quantum computing?")
        assert sq.query == "what is quantum computing?"

    def test_domain_defaults_to_empty_string(self):
        sq = SubQuery(query="test")
        assert sq.domain == ""

    def test_domain_can_be_set(self):
        sq = SubQuery(query="test", domain="clinical")
        assert sq.domain == "clinical"

    def test_strips_leading_trailing_whitespace(self):
        sq = SubQuery(query="  quantum computing  ")
        assert sq.query == "quantum computing"

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="SubQuery.query must not be empty"):
            SubQuery(query="")

    def test_whitespace_only_query_raises(self):
        with pytest.raises(ValueError, match="SubQuery.query must not be empty"):
            SubQuery(query="   ")

    def test_is_mutable(self):
        sq = SubQuery(query="original")
        sq.query = "modified"
        assert sq.query == "modified"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SubQuery)


# ---------------------------------------------------------------------------
# RawChunk
# ---------------------------------------------------------------------------


class TestRawChunk:
    def test_fields(self):
        rc = RawChunk(
            url="https://arxiv.org/abs/1234",
            text="some text",
            retrieved_at=_TS,
            source_type="latex",
            sub_query="quantum",
        )
        assert rc.url == "https://arxiv.org/abs/1234"
        assert rc.text == "some text"
        assert rc.retrieved_at == _TS
        assert rc.source_type == "latex"
        assert rc.sub_query == "quantum"

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="RawChunk.url must not be empty"):
            RawChunk(url="", text="content", retrieved_at=_TS, source_type="s", sub_query="q")

    def test_whitespace_url_raises(self):
        with pytest.raises(ValueError, match="RawChunk.url must not be empty"):
            RawChunk(url="   ", text="content", retrieved_at=_TS, source_type="s", sub_query="q")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="RawChunk.text must not be empty"):
            RawChunk(url="https://example.com", text="", retrieved_at=_TS, source_type="s", sub_query="q")

    def test_whitespace_text_raises(self):
        with pytest.raises(ValueError, match="RawChunk.text must not be empty"):
            RawChunk(url="https://example.com", text="  ", retrieved_at=_TS, source_type="s", sub_query="q")

    def test_is_mutable(self):
        rc = RawChunk(url="https://example.com", text="t", retrieved_at=_TS, source_type="s", sub_query="q")
        rc.text = "updated"
        assert rc.text == "updated"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(RawChunk)

    def test_has_no_chunk_index_field(self):
        fields = {f.name for f in dataclasses.fields(RawChunk)}
        assert "chunk_index" not in fields


# ---------------------------------------------------------------------------
# VerifiedChunk
# ---------------------------------------------------------------------------


class TestVerifiedChunk:
    def test_fields_accessible(self):
        c = _make_chunk()
        assert c.url == "https://example.com"
        assert c.text == "hello"
        assert c.retrieved_at == _TS
        assert c.source_type == "abstract"
        assert c.sub_query == "foo"
        assert c.chunk_index == 0
        assert len(c.hash) == 64  # sha256 hex digest

    def test_is_frozen(self):
        c = _make_chunk()
        with pytest.raises(FrozenInstanceError):
            c.text = "mutate"  # type: ignore[misc]

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(VerifiedChunk)

    def test_hash_matches_expected(self):
        expected = hash_chunk("https://example.com", "hello", _TS, "sha256")
        assert _make_chunk().hash == expected

    @pytest.mark.parametrize("chunk_index", [0, 1, 99])
    def test_chunk_index_stored(self, chunk_index: int):
        c = _make_chunk(chunk_index=chunk_index)
        assert c.chunk_index == chunk_index

    def test_short_hash_is_first_eight_chars(self):
        c = _make_chunk()
        assert c.short_hash == c.hash[:8]

    def test_short_hash_length(self):
        c = _make_chunk()
        assert len(c.short_hash) == 8


# ---------------------------------------------------------------------------
# AgentFindings
# ---------------------------------------------------------------------------


class TestAgentFindings:
    def test_valid_construction(self):
        chunks = [_make_chunk()]
        f = _make_findings(chunks)
        assert f.agent_type == "arxiv"
        assert f.query == "q"
        assert len(f.chunks) == 1

    def test_merkle_root_matches_tree(self):
        chunks = [_make_chunk(url=f"https://example.com/{i}", chunk_index=i) for i in range(3)]
        f = _make_findings(chunks)
        assert f.merkle_root == f.merkle_tree.root

    def test_wrong_merkle_root_raises(self):
        chunks = [_make_chunk()]
        tree = build_merkle_tree([c.hash for c in chunks], "sha256")
        with pytest.raises(ValueError, match="merkle_root mismatch"):
            AgentFindings(
                agent_type="arxiv",
                query="q",
                chunks=tuple(chunks),
                merkle_root="badhash",
                merkle_tree=tree,
            )

    def test_is_frozen(self):
        chunks = [_make_chunk()]
        f = _make_findings(chunks)
        with pytest.raises(FrozenInstanceError):
            f.agent_type = "other"  # type: ignore[misc]

    def test_chunks_stored_as_tuple(self):
        chunks = [_make_chunk()]
        f = _make_findings(chunks)
        assert isinstance(f.chunks, tuple)

    def test_multiple_chunks(self):
        chunks = [
            _make_chunk(url=f"https://example.com/{i}", text=f"text {i}", chunk_index=i)
            for i in range(5)
        ]
        f = _make_findings(chunks)
        assert len(f.chunks) == 5

    def test_chunk_count_matches_len(self):
        chunks = [_make_chunk(url=f"https://example.com/{i}", chunk_index=i) for i in range(3)]
        f = _make_findings(chunks)
        assert f.chunk_count == 3

    def test_chunk_count_zero_for_empty(self):
        tree = build_merkle_tree([], "sha256")
        f = AgentFindings(
            agent_type="arxiv", query="q", chunks=(), merkle_root="", merkle_tree=tree
        )
        assert f.chunk_count == 0

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(AgentFindings)


# ---------------------------------------------------------------------------
# CertifiedReport
# ---------------------------------------------------------------------------


class TestCertifiedReport:
    def _make(self, query: str = "q", report: str = "r") -> CertifiedReport:
        forest = ForestTree(algorithm="sha256")
        chunk = _make_chunk()
        return CertifiedReport(
            original_query=query,
            report=report,
            forest_tree=forest,
            chunks=(chunk,),
        )

    def test_fields_accessible(self):
        cr = self._make(query="test q", report="test r")
        assert cr.original_query == "test q"
        assert cr.report == "test r"
        assert isinstance(cr.forest_tree, ForestTree)
        assert len(cr.chunks) == 1

    def test_is_frozen(self):
        cr = self._make()
        with pytest.raises(FrozenInstanceError):
            cr.report = "mutate"  # type: ignore[misc]

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(CertifiedReport)

    def test_empty_chunks_allowed(self):
        forest = ForestTree(algorithm="sha256")
        cr = CertifiedReport(
            original_query="q", report="r", forest_tree=forest, chunks=()
        )
        assert cr.chunks == ()
