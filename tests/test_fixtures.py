"""
Validate that current code reproduces the reference fixture outputs.

Run with:
    python -m pytest tests/test_fixtures.py -v

Prerequisites:
    python -m tests.generate_fixtures
"""

import os
import sys
import json
import pytest
import torch
import numpy as np
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")

# Tolerances for floating point comparisons
ATOL = 1e-5
RTOL = 1e-4


# ─── Helpers ──────────────────────────────────────────────────────────

def load_tensor(name):
    return torch.load(os.path.join(FIXTURES_DIR, f"{name}.pt"), weights_only=False)


def load_numpy(name):
    return np.load(os.path.join(FIXTURES_DIR, f"{name}.npy"))


def load_json(name):
    with open(os.path.join(FIXTURES_DIR, f"{name}.json")) as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="module")
def metadata():
    return load_json("metadata")


@pytest.fixture(scope="module")
def seed(metadata):
    return metadata["seed"]


# ─── 1. Graph Generation ─────────────────────────────────────────────

class TestGraphGeneration:
    """Verify graph generation produces identical outputs with same seed."""

    def test_erdos_renyi_generation(self, metadata):
        from loaders.data_generator import (
            generate_erdos_renyi_netx,
            noise_erdos_renyi,
            adjacency_matrix_to_tensor_representation,
        )

        set_seed(metadata["seed"])
        n = metadata["n_vertices"]
        p = metadata["edge_density"]
        noise_level = metadata["noise"]

        ref_A = load_tensor("graphs_A")
        ref_B = load_tensor("graphs_B")
        ref_W = load_tensor("adjacencies_W")
        ref_W_noise = load_tensor("adjacencies_W_noise")
        ref_perms = load_tensor("permutations")

        for i in range(metadata["n_examples"]):
            g, W, _ = generate_erdos_renyi_netx(p, n)
            W_noise = noise_erdos_renyi(g, W, noise_level, p)
            B = adjacency_matrix_to_tensor_representation(W)
            B_noise = adjacency_matrix_to_tensor_representation(W_noise)

            perm = np.random.permutation(n)
            perm_mat = torch.zeros(n, n)
            for j in range(n):
                perm_mat[j, perm[j]] = 1.0

            torch.testing.assert_close(W, ref_W[i], atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(W_noise, ref_W_noise[i], atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(B, ref_A[i], atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(B_noise, ref_B[i], atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(perm_mat, ref_perms[i], atol=ATOL, rtol=RTOL)

    def test_adjacency_shapes(self, metadata):
        ref_W = load_tensor("adjacencies_W")
        n = metadata["n_vertices"]
        assert ref_W.shape == (metadata["n_examples"], n, n)

    def test_adjacency_symmetric(self):
        ref_W = load_tensor("adjacencies_W")
        for i in range(len(ref_W)):
            torch.testing.assert_close(ref_W[i], ref_W[i].T, atol=ATOL, rtol=RTOL)

    def test_adjacency_binary(self):
        ref_W = load_tensor("adjacencies_W")
        assert torch.all((ref_W == 0) | (ref_W == 1))

    def test_permutation_valid(self, metadata):
        ref_perms = load_tensor("permutations")
        n = metadata["n_vertices"]
        for i in range(len(ref_perms)):
            # Each row has exactly one 1
            assert torch.allclose(ref_perms[i].sum(dim=1), torch.ones(n))
            # Each col has exactly one 1
            assert torch.allclose(ref_perms[i].sum(dim=0), torch.ones(n))


# ─── 2. Tensor Representation ────────────────────────────────────────

class TestTensorRepresentation:
    """Verify tensor representation functions."""

    def test_standard_repr(self, metadata):
        from loaders.data_generator import adjacency_matrix_to_tensor_representation

        set_seed(metadata["seed"] + 1)
        ref_W = load_tensor("adjacencies_W")
        ref_repr = load_tensor("tensor_repr_standard")

        for i in range(min(3, len(ref_W))):
            B = adjacency_matrix_to_tensor_representation(ref_W[i])
            torch.testing.assert_close(B, ref_repr[i], atol=ATOL, rtol=RTOL)

    def test_standard_repr_channel0_is_adjacency(self):
        ref_repr = load_tensor("tensor_repr_standard")
        ref_W = load_tensor("adjacencies_W")
        for i in range(len(ref_repr)):
            torch.testing.assert_close(
                ref_repr[i][0], ref_W[i], atol=ATOL, rtol=RTOL
            )

    def test_standard_repr_channel1_diagonal(self, metadata):
        """Channel 1 diagonal should be i/n for node i."""
        ref_repr = load_tensor("tensor_repr_standard")
        n = metadata["n_vertices"]
        expected_diag = torch.tensor([i / n for i in range(n)], dtype=torch.float)
        for i in range(len(ref_repr)):
            actual_diag = ref_repr[i][1].diag()
            torch.testing.assert_close(actual_diag, expected_diag, atol=ATOL, rtol=RTOL)

    def test_ind_repr(self, metadata):
        from loaders.data_generator import adjacency_matrix_to_tensor_representation_ind

        set_seed(metadata["seed"] + 1)
        ref_W = load_tensor("adjacencies_W")
        ref_repr_ind = load_tensor("tensor_repr_ind")
        n = metadata["n_vertices"]

        for i in range(min(3, len(ref_W))):
            ind = np.random.permutation(n)
            W_2ch = torch.zeros(2, n, n)
            W_2ch[0] = ref_W[i]
            B_ind = adjacency_matrix_to_tensor_representation_ind(W_2ch, ind)
            torch.testing.assert_close(B_ind, ref_repr_ind[i], atol=ATOL, rtol=RTOL)


# ─── 3. Data Relabeling ──────────────────────────────────────────────

class TestDataRelabeling:
    """Verify make_data_from_ind_label produces same outputs."""

    def test_relabeling(self, metadata):
        from loaders.data_generator import make_data_from_ind_label

        graphs_A = load_tensor("graphs_A")
        graphs_B = load_tensor("graphs_B")
        perms = load_tensor("permutations")
        ref_relabeled = load_tensor("relabeled_data")
        ref_ind1 = load_numpy("relabel_ind1")
        ref_ind2 = load_numpy("relabel_ind2")

        n_samples = min(5, len(graphs_A))
        data = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(n_samples)]
        ind_pairs = [(ref_ind1[i], ref_ind2[i]) for i in range(n_samples)]

        relabeled = make_data_from_ind_label(data, ind_pairs)

        assert len(relabeled) == len(ref_relabeled)
        for i in range(len(relabeled)):
            for j in range(3):  # graph_A, graph_B, perm
                torch.testing.assert_close(
                    relabeled[i][j], ref_relabeled[i][j],
                    atol=ATOL, rtol=RTOL,
                )

    def test_relabeling_preserves_labels(self):
        """Relabeling should not change the permutation labels."""
        perms = load_tensor("permutations")
        ref_relabeled = load_tensor("relabeled_data")
        for i in range(len(ref_relabeled)):
            torch.testing.assert_close(
                ref_relabeled[i][2], perms[i], atol=ATOL, rtol=RTOL
            )


# ─── 4. Collate Function ─────────────────────────────────────────────

class TestCollateFunction:
    """Verify the DataLoader collate function."""

    def test_collate(self):
        from loaders import collate_fn

        graphs_A = load_tensor("graphs_A")
        graphs_B = load_tensor("graphs_B")
        perms = load_tensor("permutations")
        ref_input1 = load_tensor("collate_input1")
        ref_input2 = load_tensor("collate_input2")
        ref_target = load_tensor("collate_target")

        samples = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(4)]
        batch = collate_fn(samples)

        torch.testing.assert_close(batch[0]["input"], ref_input1, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(batch[1]["input"], ref_input2, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(batch[2], ref_target, atol=ATOL, rtol=RTOL)

    def test_collate_shapes(self, metadata):
        ref_input1 = load_tensor("collate_input1")
        n = metadata["n_vertices"]
        assert ref_input1.shape == (4, 2, n, n)


# ─── 5. Masking ──────────────────────────────────────────────────────

class TestMasking:
    """Verify masking_noseed zeros out the positional channel."""

    def test_masking_noseed(self):
        from loaders.load_utils import masking_noseed

        ref_before = load_tensor("masking_before")
        ref_after = load_tensor("masking_after")

        test_graph = ref_before.clone()
        masking_noseed(test_graph)
        torch.testing.assert_close(test_graph, ref_after, atol=ATOL, rtol=RTOL)

    def test_masking_zeros_channel1(self):
        ref_after = load_tensor("masking_after")
        assert torch.all(ref_after[1] == 0)

    def test_masking_preserves_channel0(self):
        ref_before = load_tensor("masking_before")
        ref_after = load_tensor("masking_after")
        torch.testing.assert_close(ref_before[0], ref_after[0], atol=ATOL, rtol=RTOL)


# ─── 6. Metrics ──────────────────────────────────────────────────────

class TestMetrics:
    """Verify metric functions produce identical results."""

    def test_accuracy_max_aggregate(self, metadata):
        from toolbox.metrics import accuracy_max

        set_seed(metadata["seed"] + 3)
        scores = load_tensor("metrics_scores")
        n = metadata["n_vertices"]
        bs = scores.shape[0]
        labels = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)
        ref = load_json("metrics_accuracy")

        acc, total = accuracy_max(scores, labels, aggregate_score=True)
        assert int(acc) == ref["acc"]
        assert int(total) == ref["total"]

    def test_accuracy_max_per_sample(self, metadata):
        from toolbox.metrics import accuracy_max

        scores = load_tensor("metrics_scores")
        n = metadata["n_vertices"]
        bs = scores.shape[0]
        labels = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)
        ref = load_json("metrics_accuracy")

        per_sample = accuracy_max(scores, labels, aggregate_score=False)
        np.testing.assert_allclose(per_sample, ref["per_sample"], atol=ATOL)

    def test_get_ranking_no_faq(self):
        from toolbox.metrics import get_ranking

        weight = load_numpy("metrics_weight")
        g1 = load_numpy("metrics_g1")
        g2 = load_numpy("metrics_g2")
        ref_order = load_numpy("metrics_ranking_order")
        ref_col_ind = load_numpy("metrics_ranking_col_ind")

        row_ordering, col_ind = get_ranking(weight, g1, g2, use_faq=False)
        np.testing.assert_array_equal(row_ordering, ref_order)
        np.testing.assert_array_equal(col_ind, ref_col_ind)

    def test_get_ranking_with_faq(self):
        from toolbox.metrics import get_ranking

        weight = load_numpy("metrics_weight")
        g1 = load_numpy("metrics_g1")
        g2 = load_numpy("metrics_g2")
        ref_order = load_numpy("metrics_ranking_order_faq")
        ref_col_ind = load_numpy("metrics_ranking_col_ind_faq")

        row_ordering, col_ind = get_ranking(weight, g1, g2, use_faq=True)
        np.testing.assert_array_equal(row_ordering, ref_order)
        np.testing.assert_array_equal(col_ind, ref_col_ind)

    def test_get_perm(self):
        from toolbox.metrics import get_perm

        ind0 = load_numpy("metrics_get_perm_ind0")
        ind1 = load_numpy("metrics_get_perm_ind1")
        ref_perm = load_numpy("metrics_get_perm_result")

        perm = get_perm((ind0, ind1))
        np.testing.assert_array_equal(perm, ref_perm)


# ─── 7. Model Forward Pass ───────────────────────────────────────────

class TestModelForward:
    """Verify model forward pass produces identical outputs."""

    def _load_model(self):
        from models import get_model, get_siamese

        cfg = load_json("model_config")
        state_dict = load_tensor("model_state_dict")
        node_embedder = get_model(cfg)
        siamese = get_siamese(node_embedder)
        siamese.load_state_dict(state_dict)
        siamese.eval()
        return siamese

    def test_forward_raw_scores(self):
        siamese = self._load_model()
        input1 = {"input": load_tensor("model_input1")}
        input2 = {"input": load_tensor("model_input2")}
        ref_scores = load_tensor("model_raw_scores")

        with torch.no_grad():
            raw_scores = siamese(input1, input2)

        torch.testing.assert_close(raw_scores, ref_scores, atol=ATOL, rtol=RTOL)

    def test_forward_verbose(self):
        siamese = self._load_model()
        input1 = {"input": load_tensor("model_input1")}
        input2 = {"input": load_tensor("model_input2")}
        ref_emb1 = load_tensor("model_emb1")
        ref_emb2 = load_tensor("model_emb2")

        with torch.no_grad():
            _, emb1, emb2 = siamese.forward_with_features(input1, input2)

        torch.testing.assert_close(emb1, ref_emb1, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(emb2, ref_emb2, atol=ATOL, rtol=RTOL)

    def test_loss_value(self):
        siamese = self._load_model()
        input1 = {"input": load_tensor("model_input1")}
        input2 = {"input": load_tensor("model_input2")}
        target = load_tensor("collate_target")[:3]  # same 3 samples
        ref_loss = load_json("model_loss")

        with torch.no_grad():
            raw_scores = siamese(input1, input2)
        loss = torch.nn.CrossEntropyLoss(reduction="mean")(raw_scores, target)

        np.testing.assert_allclose(loss.item(), ref_loss["loss"], atol=1e-3, rtol=1e-3)

    def test_score_shapes(self, metadata):
        ref_scores = load_tensor("model_raw_scores")
        n = metadata["n_vertices"]
        assert ref_scores.shape == (3, n, n)


# ─── 8. all_ind Inference ────────────────────────────────────────────

class TestAllInd:
    """Verify all_ind produces identical index predictions."""

    def test_all_ind_indices(self, metadata):
        from models import get_model, get_siamese
        from loaders import siamese_loader
        from loaders.data_generator import all_ind

        cfg = load_json("model_config")
        state_dict = load_tensor("all_ind_model_state")
        node_embedder = get_model(cfg)
        siamese = get_siamese(node_embedder)
        siamese.load_state_dict(state_dict)
        siamese.eval()

        graphs_A = load_tensor("graphs_A")
        graphs_B = load_tensor("graphs_B")
        perms = load_tensor("permutations")
        data = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(5)]
        loader = siamese_loader(data, batch_size=2, shuffle=False)

        ind_data, nce, faq = all_ind(
            loader, siamese, "cpu",
            compute_nce=True,
            compute_faq=True,
            verbose=False,
            size_seed=0,
        )

        ref_ind1 = load_numpy("all_ind_ind1")
        ref_ind2 = load_numpy("all_ind_ind2")
        ref_nce = load_numpy("all_ind_nce")

        for i in range(len(ind_data)):
            np.testing.assert_array_equal(ind_data[i][0], ref_ind1[i])
            np.testing.assert_array_equal(ind_data[i][1], ref_ind2[i])

        np.testing.assert_allclose(nce, ref_nce, atol=ATOL, rtol=RTOL)

    def test_all_ind_nce_nonnegative(self):
        ref_nce = load_numpy("all_ind_nce")
        assert np.all(ref_nce >= 0)


# ─── Smoke Test ───────────────────────────────────────────────────────

class TestFixturesExist:
    """Verify all expected fixture files exist."""

    EXPECTED_FILES = [
        "metadata.json",
        "graphs_A.pt",
        "graphs_B.pt",
        "adjacencies_W.pt",
        "adjacencies_W_noise.pt",
        "permutations.pt",
        "tensor_repr_standard.pt",
        "tensor_repr_ind.pt",
        "relabeled_data.pt",
        "relabel_ind1.npy",
        "relabel_ind2.npy",
        "collate_input1.pt",
        "collate_input2.pt",
        "collate_target.pt",
        "masking_before.pt",
        "masking_after.pt",
        "metrics_scores.pt",
        "metrics_accuracy.json",
        "metrics_weight.npy",
        "metrics_g1.npy",
        "metrics_g2.npy",
        "metrics_ranking_order.npy",
        "metrics_ranking_col_ind.npy",
        "metrics_ranking_order_faq.npy",
        "metrics_ranking_col_ind_faq.npy",
        "metrics_get_perm_ind0.npy",
        "metrics_get_perm_ind1.npy",
        "metrics_get_perm_result.npy",
        "model_config.json",
        "model_state_dict.pt",
        "model_input1.pt",
        "model_input2.pt",
        "model_raw_scores.pt",
        "model_emb1.pt",
        "model_emb2.pt",
        "model_loss.json",
        "all_ind_model_state.pt",
        "all_ind_ind1.npy",
        "all_ind_ind2.npy",
        "all_ind_nce.npy",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_fixture_exists(self, filename):
        path = os.path.join(FIXTURES_DIR, filename)
        assert os.path.exists(path), f"Missing fixture: {filename}"
