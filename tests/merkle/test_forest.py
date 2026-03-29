import hashlib

import pytest

from mara.merkle.forest import ForestTree, build_forest_tree
from mara.merkle.tree import build_merkle_tree, combine_hashes


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class TestBuildForestTree:
    def test_empty_input_returns_empty_forest(self):
        forest = build_forest_tree([], "sha256")
        assert forest.root == ""
        assert forest.agent_roots == {}

    def test_empty_input_carries_algorithm(self):
        forest = build_forest_tree([], "sha256")
        assert forest.algorithm == "sha256"

    def test_single_agent_root_equals_subtree_root(self):
        sub_root = sha256("only-agent")
        forest = build_forest_tree([("arxiv", sub_root)], "sha256")
        assert forest.root == sub_root

    def test_single_agent_roots_mapping(self):
        sub_root = sha256("only-agent")
        forest = build_forest_tree([("arxiv", sub_root)], "sha256")
        assert forest.agent_roots == {"arxiv": sub_root}

    def test_two_agents_root_matches_manual_tree(self):
        r_arxiv = sha256("arxiv-root")
        r_web = sha256("web-root")
        # Alphabetical: arxiv < web
        expected_root = build_merkle_tree([r_arxiv, r_web], "sha256").root
        forest = build_forest_tree([("arxiv", r_arxiv), ("web", r_web)], "sha256")
        assert forest.root == expected_root

    def test_leaf_order_is_alphabetical(self):
        r_web = sha256("web-root")
        r_arxiv = sha256("arxiv-root")
        # Reversed input order — root must be the same as alphabetical order
        forest_reversed = build_forest_tree([("web", r_web), ("arxiv", r_arxiv)], "sha256")
        forest_sorted = build_forest_tree([("arxiv", r_arxiv), ("web", r_web)], "sha256")
        assert forest_reversed.root == forest_sorted.root

    def test_order_invariant_with_five_agents(self):
        agents = [(name, sha256(name)) for name in ["web", "pubmed", "arxiv", "s2", "core"]]
        import random
        shuffled = agents[:]
        random.shuffle(shuffled)
        f1 = build_forest_tree(agents, "sha256")
        f2 = build_forest_tree(shuffled, "sha256")
        assert f1.root == f2.root

    def test_agent_roots_dict_is_sorted(self):
        agents = [("web", sha256("w")), ("arxiv", sha256("a")), ("pubmed", sha256("p"))]
        forest = build_forest_tree(agents, "sha256")
        assert list(forest.agent_roots.keys()) == sorted(forest.agent_roots.keys())

    def test_agent_roots_values_correct(self):
        r_arxiv = sha256("arxiv-root")
        r_s2 = sha256("s2-root")
        forest = build_forest_tree([("s2", r_s2), ("arxiv", r_arxiv)], "sha256")
        assert forest.agent_roots["arxiv"] == r_arxiv
        assert forest.agent_roots["s2"] == r_s2

    def test_algorithm_stored_on_forest(self):
        forest = build_forest_tree([("arxiv", sha256("x"))], "sha256")
        assert forest.algorithm == "sha256"

    def test_meta_tree_stored_on_forest(self):
        sub_root = sha256("arxiv")
        forest = build_forest_tree([("arxiv", sub_root)], "sha256")
        assert forest.meta_tree is not None
        assert forest.meta_tree.root == forest.root

    def test_root_is_alias_for_meta_tree_root(self):
        agents = [("arxiv", sha256("a")), ("web", sha256("w"))]
        forest = build_forest_tree(agents, "sha256")
        assert forest.root == forest.meta_tree.root

    def test_duplicate_agent_type_raises(self):
        sub_root = sha256("x")
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            build_forest_tree([("arxiv", sub_root), ("arxiv", sha256("y"))], "sha256")

    def test_duplicate_not_silently_dropped(self):
        # Ensure the second arxiv entry's root is not silently ignored
        r1, r2 = sha256("root1"), sha256("root2")
        with pytest.raises(ValueError):
            build_forest_tree([("arxiv", r1), ("arxiv", r2)], "sha256")
