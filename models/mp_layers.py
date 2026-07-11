"""Message-passing GNN layers for graph alignment.

Provides building blocks that accept the same dense (batch, 2, n, n) input
format used by the rest of the codebase and produce node embeddings of shape
(batch, features, n), matching the interface expected by the Siamese wrappers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv

from models.layers import PositionalEncoding, Diag_sum


# ---------------------------------------------------------------------------
# Dense-to-sparse conversion (pure PyTorch, no compiled extensions)
# ---------------------------------------------------------------------------

def _batched_dense_to_sparse(adj: torch.Tensor):
    """Convert a batch of dense adjacency matrices to a single COO edge_index.

    Uses only core PyTorch ops (no compiled PyG extensions) to avoid
    segfaults from extension/CUDA version mismatches on clusters.

    Args:
        adj: (B, N, N) batch of adjacency matrices.

    Returns:
        edge_index: (2, total_edges) with node indices offset per sample.
    """
    B, N, _ = adj.shape
    # Find all non-zero entries across the batch
    batch_idx, row, col = torch.nonzero(adj, as_tuple=True)
    # Offset node indices so each sample occupies its own range [i*N, (i+1)*N)
    offset = batch_idx * N
    edge_index = torch.stack([row + offset, col + offset], dim=0)
    return edge_index


class DenseToSparse(nn.Module):
    """Convert dense (batch, 2, n, n) tensors to PyG sparse format.

    Channel 0 of the input is the adjacency matrix; channel 1 carries a
    diagonal positional encoding (value i/n at position (i,i)).

    Returns (x, edge_index, batch_index) where:
      - x:           (total_nodes, in_features) node feature matrix
      - edge_index:  (2, total_edges) COO edge indices
      - batch_index: (total_nodes,) mapping each node to its sample
    """

    def __init__(self, in_features: int):
        super().__init__()
        # Initial node features: degree (from ch0) + positional scalar (from ch1) → 2 dims
        self.proj = nn.Linear(2, in_features)

    def forward(self, inp: torch.Tensor):
        # inp: (B, 2, N, N)
        B, _, N, _ = inp.shape
        adj = inp[:, 0, :, :]          # (B, N, N)

        # Equivariant node features:
        # 1) Node degree from adjacency (permutation equivariant)
        degree = adj.sum(dim=-1)  # (B, N)
        # 2) Diagonal of channel 1: positional / matching encoding
        pos = torch.diagonal(inp[:, 1, :, :], dim1=-2, dim2=-1)  # (B, N)
        # Concatenate and project
        node_feat = torch.stack([degree, pos], dim=-1)  # (B, N, 2)
        node_feat = self.proj(node_feat)  # (B, N, in_features)

        # Build batched edge_index (pure PyTorch, no compiled extensions)
        edge_index = _batched_dense_to_sparse(adj)  # (2, total_edges)

        # Flatten node features
        x = node_feat.reshape(B * N, -1)  # (B*N, in_features)

        # Batch index
        batch_index = torch.arange(B, device=inp.device).repeat_interleave(N)

        return x, edge_index, batch_index


# ---------------------------------------------------------------------------
# Message-passing block
# ---------------------------------------------------------------------------

_CONV_REGISTRY = {
    "GCN": GCNConv,
    "GAT": GATConv,
    "GIN": GINConv,
    "SAGE": SAGEConv,
}


def _make_conv(conv_type: str, in_dim: int, out_dim: int, num_heads: int = 4):
    """Instantiate a PyG convolution layer by name."""
    if conv_type == "GAT":
        assert out_dim % num_heads == 0, (
            f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        )
        return GATConv(in_dim, out_dim // num_heads, heads=num_heads, concat=True)
    elif conv_type == "GIN":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        return GINConv(mlp)
    elif conv_type in ("GCN", "SAGE"):
        cls = _CONV_REGISTRY[conv_type]
        return cls(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type!r}. "
                         f"Available: {list(_CONV_REGISTRY)}")


class MPBlock(nn.Module):
    """Single message-passing layer with LayerNorm, activation, and residual."""

    def __init__(self, in_dim: int, out_dim: int, conv_type: str = "GCN",
                 num_heads: int = 4):
        super().__init__()
        self.conv = _make_conv(conv_type, in_dim, out_dim, num_heads)
        self.norm = nn.LayerNorm(out_dim)
        self.residual = (in_dim == out_dim)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        if self.residual:
            h = h + x
        return h


# ---------------------------------------------------------------------------
# Complete node embedding module
# ---------------------------------------------------------------------------

class MPNodeEmbedding(nn.Module):
    """Full MP-GNN node embedding: dense→sparse→MP layers→dense.

    Receives the raw (batch, 2, n, n) tensor and outputs (batch, features, n)
    node embeddings, optionally concatenated with positional encoding.
    """

    def __init__(self, original_features_num: int, num_blocks: int,
                 in_features: int, conv_type: str = "GCN",
                 num_heads: int = 4):
        super().__init__()
        self.in_features = in_features

        # Dense to sparse conversion with feature projection
        self.dense_to_sparse = DenseToSparse(in_features)

        # Stack of MP layers
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(MPBlock(in_features, in_features, conv_type, num_heads))

        # Positional encoding branch (mirrors node_embedding_node_pos)
        self.diag_sum = Diag_sum()
        self.pos_enc = PositionalEncoding(in_features)

    def forward(self, inp: torch.Tensor):
        # inp: (B, 2, N, N)
        B, _, N, _ = inp.shape

        # Convert to sparse
        x, edge_index, batch_index = self.dense_to_sparse(inp)

        # Message-passing layers
        for block in self.blocks:
            x = block(x, edge_index)

        # Reshape back to dense: (B*N, F) → (B, N, F) → (B, F, N)
        x = x.view(B, N, -1).permute(0, 2, 1)  # (B, F, N)

        # Positional encoding branch
        pos = self.diag_sum(inp)        # (B, N) — sum of diagonal values
        pe = self.pos_enc(pos)          # (B, F, N)

        # Concatenate MP output with positional encoding
        out = torch.cat([x, pe], dim=1)  # (B, 2*F, N)

        return out
