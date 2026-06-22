"""Tests for the qap_loop permutation fix.

Verifies that:
1. build_ind / make_data_from_ind_label does NOT permute adjacency matrices
2. LAP results are direct graph1→graph2 permutations (not corrections)
3. Using col_ind directly (not composed) gives correct QAP objectives

Run with:
    python -m pytest tests/test_qap_loop_perm.py -v
"""

import numpy as np
import torch
import pytest

from loaders.representations import adjacency_matrix_to_tensor_representation_ind
from models.pipeline import Chaining


class TestPositionalEncodingOnly:
    """adjacency_matrix_to_tensor_representation_ind must not change channel 0."""

    def test_channel0_unchanged(self):
        n = 10
        W = torch.zeros(2, n, n)
        # Random symmetric adjacency in channel 0
        adj = torch.randint(0, 2, (n, n)).float()
        adj = (adj + adj.T).clamp(max=1)
        adj.fill_diagonal_(0)
        W[0] = adj
        # Some positional encoding in channel 1
        W[1, range(n), range(n)] = torch.arange(n, dtype=torch.float) / n

        ind = np.random.permutation(n)
        B = adjacency_matrix_to_tensor_representation_ind(W, ind)

        # Channel 0 must be identical
        torch.testing.assert_close(B[0], W[0])

    def test_channel1_updated(self):
        n = 8
        W = torch.zeros(2, n, n)
        W[0] = torch.eye(n)  # dummy adjacency

        ind = np.array([3, 1, 4, 0, 2, 7, 5, 6])
        B = adjacency_matrix_to_tensor_representation_ind(W, ind)

        # Check positional encoding: B[1, ind[i], ind[i]] = i/n
        for i, j in enumerate(ind):
            assert abs(B[1, j, j].item() - i / n) < 1e-7

    def test_repeated_application_preserves_adjacency(self):
        """Simulates what happens across chaining iterations."""
        n = 6
        W = torch.zeros(2, n, n)
        adj = torch.randint(0, 2, (n, n)).float()
        adj = (adj + adj.T).clamp(max=1)
        adj.fill_diagonal_(0)
        W[0] = adj
        original_adj = adj.clone()

        # Simulate 3 iterations of build_ind
        current = W
        for _ in range(3):
            ind = np.random.permutation(n)
            current = adjacency_matrix_to_tensor_representation_ind(current, ind)
            # Adjacency never changes
            torch.testing.assert_close(current[0], original_adj)


class TestPermutationNotComposed:
    """Verify that using col_ind directly gives correct QAP objectives,
    while composing gives wrong results."""

    def _make_qap_instance(self, n, seed=42):
        """Create a QAP instance where the optimal permutation is known."""
        rng = np.random.RandomState(seed)
        # A is a random adjacency
        A = (rng.rand(n, n) > 0.5).astype(float)
        A = ((A + A.T) > 0).astype(float)
        np.fill_diagonal(A, 0)

        # True permutation
        true_perm = rng.permutation(n)

        # B = A permuted by true_perm (so QAP objective is maximized at true_perm)
        B = A[true_perm, :][:, true_perm]

        return A, B, true_perm

    def test_direct_perm_gives_correct_objective(self):
        """If LAP finds the inverse permutation, using it directly is correct."""
        n = 10
        A, B, true_perm = self._make_qap_instance(n)
        # B = A[true_perm][:,true_perm], so B[inv_perm][:,inv_perm] = A
        inv_perm = np.argsort(true_perm)

        def qap_obj(perm):
            return (A * B[perm, :][:, perm]).sum()

        # Direct use of inverse: correct
        obj_direct = qap_obj(inv_perm)
        expected = (A * A).sum()
        assert abs(obj_direct - expected) < 1e-10

    def test_composition_is_wrong(self):
        """Composing identical permutations gives π² ≠ π (unless π = identity)."""
        n = 10
        rng = np.random.RandomState(123)
        perm = rng.permutation(n)

        # Composition: perm[perm] = π²
        composed = perm[perm]

        # π² ≠ π for a random permutation (with very high probability)
        assert not np.array_equal(composed, perm), \
            "Random permutation squared should differ from itself"

    def test_old_vs_new_on_known_answer(self):
        """Simulate old (composed) vs new (direct) qap_loop behavior."""
        n = 8
        A, B, true_perm = self._make_qap_instance(n)
        # The optimal LAP answer is inv_perm: B[inv_perm][:,inv_perm] = A
        inv_perm = np.argsort(true_perm)

        def qap_obj(perm):
            return (A * B[perm, :][:, perm]).sum()

        optimal_obj = qap_obj(inv_perm)

        # Suppose LAP returns inv_perm in both iterations
        # (as would happen if the GNN is good and positional hints work)

        # NEW behavior: use col_ind directly
        col_ind_0 = inv_perm.copy()
        col_ind_1 = inv_perm.copy()  # same answer again with better hints
        new_obj_0 = qap_obj(col_ind_0)
        new_obj_1 = qap_obj(col_ind_1)
        assert abs(new_obj_0 - optimal_obj) < 1e-10
        assert abs(new_obj_1 - optimal_obj) < 1e-10

        # OLD behavior: compose permutations
        composed = np.arange(n)
        composed = composed[col_ind_0]  # = inv_perm
        composed = composed[col_ind_1]  # = inv_perm[inv_perm] = π⁻²
        old_obj = qap_obj(composed)

        # Old gives π⁻² which is wrong (unless π is involution)
        # The composed permutation should differ from the direct one
        assert not np.array_equal(composed, inv_perm), \
            "Composed should differ from direct (unless permutation is involution)"
        # And the direct approach gives optimal objective
        assert new_obj_1 >= old_obj, \
            f"Direct perm ({new_obj_1}) should be >= composed ({old_obj})"


class TestUpdatePositionalEncoding:
    """Verify _update_positional_encoding sets matched nodes to same rank."""

    def test_matched_nodes_share_rank(self):
        n = 8
        # Build a sample: (graph1_tensor, graph2_tensor)
        g1 = torch.zeros(2, n, n)
        g1[0] = torch.eye(n)  # dummy adjacency
        g2 = torch.zeros(2, n, n)
        g2[0] = torch.eye(n)
        sample = (g1, g2)

        perm = np.array([3, 0, 1, 2, 7, 6, 5, 4])
        new_g1, new_g2 = Chaining._update_positional_encoding(sample, perm)

        # Graph1 node i should have rank i/n
        for i in range(n):
            assert abs(new_g1[1, i, i].item() - i / n) < 1e-7

        # Graph2 node perm[i] should have rank i/n (same as its match)
        for i in range(n):
            assert abs(new_g2[1, perm[i], perm[i]].item() - i / n) < 1e-7

    def test_adjacency_preserved(self):
        n = 6
        g1 = torch.zeros(2, n, n)
        adj = torch.randint(0, 2, (n, n)).float()
        adj = (adj + adj.T).clamp(max=1)
        adj.fill_diagonal_(0)
        g1[0] = adj
        g2 = torch.zeros(2, n, n)
        g2[0] = torch.eye(n)
        sample = (g1, g2)

        perm = np.random.permutation(n)
        new_g1, new_g2 = Chaining._update_positional_encoding(sample, perm)

        torch.testing.assert_close(new_g1[0], adj)
        torch.testing.assert_close(new_g2[0], torch.eye(n))

    def test_label_preserved(self):
        n = 4
        g1 = torch.zeros(2, n, n)
        g2 = torch.zeros(2, n, n)
        label = torch.tensor([2, 0, 3, 1])
        sample = (g1, g2, label)

        perm = np.arange(n)
        result = Chaining._update_positional_encoding(sample, perm)

        assert len(result) == 3
        torch.testing.assert_close(result[2], label)
