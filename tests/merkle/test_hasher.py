import hashlib
import json

import pytest

from mara.merkle.hasher import canonical_serialise, hash_chunk


class TestCanonicalSerialise:
    def test_output_is_bytes(self):
        result = canonical_serialise("https://example.com", "hello", "2024-01-01T00:00:00Z")
        assert isinstance(result, bytes)

    def test_deterministic_key_order(self):
        # Same fields, different insertion order in the payload dict —
        # canonical_serialise always sorts keys so output must be identical.
        a = canonical_serialise("https://x.com", "text", "2024-01-01T00:00:00Z")
        b = canonical_serialise("https://x.com", "text", "2024-01-01T00:00:00Z")
        assert a == b

    def test_keys_are_sorted_alphabetically(self):
        result = canonical_serialise("https://example.com", "hello", "2024-01-01T00:00:00Z")
        decoded = result.decode("utf-8")
        # retrieved_at < text < url alphabetically
        assert decoded.index('"retrieved_at"') < decoded.index('"text"') < decoded.index('"url"')

    def test_no_extra_whitespace(self):
        result = canonical_serialise("u", "t", "r")
        assert b" " not in result

    def test_non_ascii_text_is_escaped(self):
        result = canonical_serialise("https://x.com", "\u4e2d\u6587", "2024-01-01T00:00:00Z")
        decoded = result.decode("ascii")  # must be valid ASCII
        assert "\\u" in decoded

    def test_changing_url_changes_output(self):
        a = canonical_serialise("https://a.com", "text", "2024-01-01T00:00:00Z")
        b = canonical_serialise("https://b.com", "text", "2024-01-01T00:00:00Z")
        assert a != b

    def test_changing_text_changes_output(self):
        a = canonical_serialise("https://x.com", "foo", "2024-01-01T00:00:00Z")
        b = canonical_serialise("https://x.com", "bar", "2024-01-01T00:00:00Z")
        assert a != b

    def test_changing_retrieved_at_changes_output(self):
        a = canonical_serialise("https://x.com", "text", "2024-01-01T00:00:00Z")
        b = canonical_serialise("https://x.com", "text", "2024-01-02T00:00:00Z")
        assert a != b

    def test_output_is_valid_json(self):
        result = canonical_serialise("https://example.com", "hello", "2024-01-01T00:00:00Z")
        parsed = json.loads(result)
        assert parsed["url"] == "https://example.com"
        assert parsed["text"] == "hello"
        assert parsed["retrieved_at"] == "2024-01-01T00:00:00Z"


class TestHashChunk:
    def test_returns_hex_string(self):
        result = hash_chunk("https://x.com", "text", "2024-01-01T00:00:00Z", "sha256")
        assert isinstance(result, str)
        # sha256 hex digest is 64 chars
        assert len(result) == 64

    def test_matches_manual_sha256(self):
        url, text, ts = "https://example.com", "hello world", "2024-01-01T12:00:00Z"
        data = canonical_serialise(url, text, ts)
        expected = hashlib.sha256(data).hexdigest()
        assert hash_chunk(url, text, ts, "sha256") == expected

    def test_md5_algorithm_passthrough(self):
        url, text, ts = "https://x.com", "t", "2024-01-01T00:00:00Z"
        data = canonical_serialise(url, text, ts)
        expected = hashlib.md5(data).hexdigest()  # noqa: S324
        assert hash_chunk(url, text, ts, "md5") == expected

    def test_deterministic(self):
        args = ("https://x.com", "same text", "2024-01-01T00:00:00Z", "sha256")
        assert hash_chunk(*args) == hash_chunk(*args)

    def test_different_fields_produce_different_hashes(self):
        base = ("https://x.com", "text", "2024-01-01T00:00:00Z", "sha256")
        h1 = hash_chunk(*base)
        h2 = hash_chunk("https://y.com", "text", "2024-01-01T00:00:00Z", "sha256")
        assert h1 != h2
