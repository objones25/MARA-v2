import hashlib

import pytest

from mara.merkle.proof import ProofStep, generate_merkle_proof, verify_merkle_proof
from mara.merkle.tree import build_merkle_tree, combine_hashes


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_tree(n: int):
    leaves = [sha256(str(i)) for i in range(n)]
    return build_merkle_tree(leaves, "sha256"), leaves


class TestGenerateMerkleProof:
    def test_empty_tree_raises(self):
        tree = build_merkle_tree([], "sha256")
        with pytest.raises(ValueError, match="empty"):
            generate_merkle_proof(tree, 0)

    def test_negative_index_raises(self):
        tree, _ = make_tree(3)
        with pytest.raises(IndexError):
            generate_merkle_proof(tree, -1)

    def test_index_out_of_range_raises(self):
        tree, _ = make_tree(3)
        with pytest.raises(IndexError):
            generate_merkle_proof(tree, 3)

    def test_single_leaf_proof_is_empty(self):
        tree, _ = make_tree(1)
        proof = generate_merkle_proof(tree, 0)
        assert proof == []

    def test_two_leaves_proof_length(self):
        tree, _ = make_tree(2)
        assert len(generate_merkle_proof(tree, 0)) == 1
        assert len(generate_merkle_proof(tree, 1)) == 1

    def test_four_leaves_proof_length(self):
        tree, _ = make_tree(4)
        for i in range(4):
            assert len(generate_merkle_proof(tree, i)) == 2

    def test_proof_steps_have_correct_positions_two_leaves(self):
        tree, _ = make_tree(2)
        proof_0 = generate_merkle_proof(tree, 0)
        assert proof_0[0].position == "right"
        proof_1 = generate_merkle_proof(tree, 1)
        assert proof_1[0].position == "left"

    def test_proof_step_is_dataclass(self):
        tree, _ = make_tree(2)
        step = generate_merkle_proof(tree, 0)[0]
        assert isinstance(step, ProofStep)
        assert isinstance(step.sibling_hash, str)
        assert step.position in ("left", "right")


class TestVerifyMerkleProof:
    def test_single_leaf_verifies(self):
        tree, leaves = make_tree(1)
        proof = generate_merkle_proof(tree, 0)
        assert verify_merkle_proof(leaves[0], proof, tree.root, "sha256") is True

    def test_two_leaves_both_verify(self):
        tree, leaves = make_tree(2)
        for i in range(2):
            proof = generate_merkle_proof(tree, i)
            assert verify_merkle_proof(leaves[i], proof, tree.root, "sha256") is True

    def test_four_leaves_all_verify(self):
        tree, leaves = make_tree(4)
        for i in range(4):
            proof = generate_merkle_proof(tree, i)
            assert verify_merkle_proof(leaves[i], proof, tree.root, "sha256") is True

    def test_five_leaves_all_verify(self):
        # Five leaves triggers odd-padding at every level
        tree, leaves = make_tree(5)
        for i in range(5):
            proof = generate_merkle_proof(tree, i)
            assert verify_merkle_proof(leaves[i], proof, tree.root, "sha256") is True

    def test_eight_leaves_all_verify(self):
        tree, leaves = make_tree(8)
        for i in range(8):
            proof = generate_merkle_proof(tree, i)
            assert verify_merkle_proof(leaves[i], proof, tree.root, "sha256") is True

    def test_tampered_leaf_hash_fails(self):
        tree, leaves = make_tree(4)
        proof = generate_merkle_proof(tree, 0)
        assert verify_merkle_proof("deadbeef" * 8, proof, tree.root, "sha256") is False

    def test_tampered_expected_root_fails(self):
        tree, leaves = make_tree(4)
        proof = generate_merkle_proof(tree, 0)
        assert verify_merkle_proof(leaves[0], proof, "deadbeef" * 8, "sha256") is False

    def test_wrong_algorithm_fails(self):
        tree, leaves = make_tree(4)
        proof = generate_merkle_proof(tree, 0)
        # Proof was built with sha256; verifying with md5 should fail
        assert verify_merkle_proof(leaves[0], proof, tree.root, "md5") is False
