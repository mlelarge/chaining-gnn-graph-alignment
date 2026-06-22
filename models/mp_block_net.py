"""Factory function for message-passing GNN node embeddings.

Returns a dict compatible with the Network computation graph used by the
Siamese wrappers.  The single ``MPNodeEmbedding`` module handles the full
dense → sparse → MP → dense pipeline internally.
"""

from __future__ import annotations

from models.layers import Identity
from models.mp_layers import MPNodeEmbedding


def node_embedding_mpgnn(original_features_num: int, num_blocks: int,
                         in_features: int, conv_type: str = "GCN",
                         num_heads: int = 4, **kwargs):
    """Build a Network-compatible dict for an MP-GNN node embedder.

    Args:
        original_features_num: Number of input channels (typically 2).
        num_blocks: Number of message-passing layers.
        in_features: Hidden dimension for node features.
        conv_type: One of "GCN", "GAT", "GIN", "SAGE".
        num_heads: Number of attention heads (GAT only).
        **kwargs: Ignored (allows shared config keys like depth_of_mlp).

    Returns:
        Dict to be wrapped by ``Network``.
    """
    return {
        'in': Identity(),
        'suffix': (MPNodeEmbedding(original_features_num, num_blocks,
                                   in_features, conv_type, num_heads), ['in']),
    }
