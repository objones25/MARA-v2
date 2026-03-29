import hashlib

import pytest

from mara.merkle.tree import MerkleTree, build_merkle_tree, combine_hashes


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class TestCombineHashes:
    def test_returns_hex_string(self):
        result = combine_hashes("aaa", "bbb", "sha256")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_matches_manual_hash(self):
        left, right = "abc", "def"
        expected = hashlib.sha256((left + right).encode("utf-8")).hexdigest()
        assert combine_hashes(left, right, "sha256") == expected

    def test_not_commutative(self):
        a = combine_hashes("left", "right", "sha256")
        b = combine_hashes("right", "left", "sha256")
        assert a != b

    def test_algorithm_passthrough(self):
        left, right = "x", "y"
        expected = hashlib.md5((left + right).encode("utf-8")).hexdigest()  # noqa: S324
        assert combine_hashes(left, right, "md5") == expected


class TestBuildMerkleTree:
    def test_empty_list_returns_empty_tree(self):
        tree = build_merkle_tree([], "sha256")
        assert tree.root == ""
        assert tree.leaves == []
        assert tree.levels == []

    def test_empty_tree_carries_algorithm(self):
        tree = build_merkle_tree([], "sha256")
        assert tree.algorithm == "sha256"

    def test_single_leaf_root_equals_leaf(self):
        leaf = sha256("only")
        tree = build_merkle_tree([leaf], "sha256")
        assert tree.root == leaf
        assert tree.leaves == [leaf]

    def test_two_leaves_root_is_combine(self):
        l0 = sha256("first")
        l1 = sha256("second")
        tree = build_merkle_tree([l0, l1], "sha256")
        assert tree.root == combine_hashes(l0, l1, "sha256")

    def test_three_leaves_last_duplicated(self):
        l0, l1, l2 = sha256("a"), sha256("b"), sha256("c")
        tree = build_merkle_tree([l0, l1, l2], "sha256")
        # Leaf level padded: [l0, l1, l2, l2]
        parent_0 = combine_hashes(l0, l1, "sha256")
        parent_1 = combine_hashes(l2, l2, "sha256")
        expected_root = combine_hashes(parent_0, parent_1, "sha256")
        assert tree.root == expected_root

    def test_eight_leaves_levels_count(self):
        leaves = [sha256(str(i)) for i in range(8)]
        tree = build_merkle_tree(leaves, "sha256")
        # 8 → 4 → 2 → 1, so levels = [leaf, L2, L3, root]
        assert len(tree.levels) == 4

    def test_leaves_preserved_in_order(self):
        leaves = [sha256(str(i)) for i in range(5)]
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.leaves == leaves

    def test_empty_string_leaf_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_merkle_tree(["", sha256("ok")], "sha256")

    def test_deterministic(self):
        leaves = [sha256(str(i)) for i in range(6)]
        assert build_merkle_tree(leaves, "sha256").root == build_merkle_tree(leaves, "sha256").root

    def test_algorithm_stored_on_tree(self):
        leaves = [sha256("x")]
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.algorithm == "sha256"
