"""
Tests for message-passing GNN architectures.

Run with:
    python -m pytest tests/test_mp_models.py -v
"""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models import get_model, get_siamese
from models.config import SiameseMode, OptimizationConfig
from models.mp_layers import DenseToSparse, MPBlock, MPNodeEmbedding


# ─── Helpers ──────────────────────────────────────────────────────────

BATCH = 4
N = 20
IN_FEATURES = 32  # small for fast tests


def _make_dummy_batch(batch_size=BATCH, n=N):
    """Create a dummy batch in the expected format: (dict, dict, labels)."""
    def _make_graph():
        adj = (torch.rand(batch_size, n, n) > 0.5).float()
        adj = (adj + adj.transpose(-1, -2)).clamp(max=1.0)  # symmetric
        adj.diagonal(dim1=-2, dim2=-1).zero_()               # no self-loops
        pos = torch.zeros(batch_size, n, n)
        for i in range(n):
            pos[:, i, i] = i / n
        inp = torch.stack([adj, pos], dim=1)  # (B, 2, N, N)
        return {"input": inp}

    g1 = _make_graph()
    g2 = _make_graph()
    labels = torch.eye(n).unsqueeze(0).expand(batch_size, -1, -1)
    return g1, g2, labels


def _make_mp_config(conv_type, num_heads=4):
    cfg = {
        "type": "node_embedding_mpgnn",
        "conv_type": conv_type,
        "num_blocks": 2,
        "in_features": IN_FEATURES,
    }
    if conv_type == "GAT":
        cfg["num_heads"] = num_heads
    return cfg


# ─── DenseToSparse ────────────────────────────────────────────────────

class TestDenseToSparse:
    def test_output_shapes(self):
        d2s = DenseToSparse(IN_FEATURES)
        inp = torch.randn(BATCH, 2, N, N)
        inp[:, 0] = (inp[:, 0] > 0).float()  # binary adjacency
        x, edge_index, batch_idx = d2s(inp)

        assert x.shape == (BATCH * N, IN_FEATURES)
        assert edge_index.shape[0] == 2
        assert batch_idx.shape == (BATCH * N,)

    def test_batch_index_values(self):
        d2s = DenseToSparse(IN_FEATURES)
        inp = torch.randn(BATCH, 2, N, N)
        inp[:, 0] = (inp[:, 0] > 0).float()
        _, _, batch_idx = d2s(inp)

        # Each sample should have exactly N nodes
        for i in range(BATCH):
            assert (batch_idx == i).sum().item() == N


# ─── MPBlock ──────────────────────────────────────────────────────────

class TestMPBlock:
    @pytest.mark.parametrize("conv_type", ["GCN", "GAT", "GIN", "SAGE"])
    def test_output_shape(self, conv_type):
        block = MPBlock(IN_FEATURES, IN_FEATURES, conv_type, num_heads=4)
        # Simple 2-node graph with one edge
        x = torch.randn(2, IN_FEATURES)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        out = block(x, edge_index)
        assert out.shape == (2, IN_FEATURES)


# ─── MPNodeEmbedding ─────────────────────────────────────────────────

class TestMPNodeEmbedding:
    @pytest.mark.parametrize("conv_type", ["GCN", "GAT", "GIN", "SAGE"])
    def test_output_shape(self, conv_type):
        emb = MPNodeEmbedding(2, num_blocks=2, in_features=IN_FEATURES,
                              conv_type=conv_type, num_heads=4)
        g1, _, _ = _make_dummy_batch()
        out = emb(g1["input"])
        # Output: (B, 2*F, N) — MP features + positional encoding
        assert out.shape == (BATCH, 2 * IN_FEATURES, N)


# ─── get_model integration ───────────────────────────────────────────

class TestGetModel:
    @pytest.mark.parametrize("conv_type", ["GCN", "GAT", "GIN", "SAGE"])
    def test_model_creation(self, conv_type):
        cfg = _make_mp_config(conv_type)
        model = get_model(cfg)
        g1, _, _ = _make_dummy_batch()
        out = model(g1)
        assert "ne/suffix" in out
        assert out["ne/suffix"].shape == (BATCH, 2 * IN_FEATURES, N)


# ─── Full Siamese forward pass ───────────────────────────────────────

class TestSiameseMP:
    @pytest.mark.parametrize("conv_type", ["GCN", "GAT", "GIN", "SAGE"])
    def test_labeled_forward(self, conv_type):
        cfg = _make_mp_config(conv_type)
        model = get_model(cfg)
        siamese = get_siamese(model, mode=SiameseMode.LABELED)
        g1, g2, labels = _make_dummy_batch()
        scores = siamese(g1, g2)
        assert scores.shape == (BATCH, N, N)

    @pytest.mark.parametrize("conv_type", ["GCN", "GIN"])
    def test_unlabeled_forward(self, conv_type):
        cfg = _make_mp_config(conv_type)
        model = get_model(cfg)
        siamese = get_siamese(model, mode=SiameseMode.UNLABELED)
        g1, g2, labels = _make_dummy_batch()
        scores = siamese(g1, g2)
        assert scores.shape == (BATCH, N, N)

    def test_training_step_labeled(self):
        cfg = _make_mp_config("GCN")
        model = get_model(cfg)
        siamese = get_siamese(model, mode=SiameseMode.LABELED)
        batch = _make_dummy_batch()
        loss = siamese.training_step(batch, 0)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_training_step_unlabeled(self):
        cfg = _make_mp_config("GCN")
        model = get_model(cfg)
        siamese = get_siamese(model, mode=SiameseMode.UNLABELED)
        batch = _make_dummy_batch()
        loss = siamese.training_step(batch, 0)
        assert loss.shape == ()
        assert torch.isfinite(loss)
