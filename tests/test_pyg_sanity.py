"""Minimal sanity checks for PyTorch Geometric on this machine.

Run with:
    python tests/test_pyg_sanity.py

Each test prints PASS/FAIL so you can diagnose cluster issues without pytest.
"""

import sys
import traceback

def test_import():
    """Can we import torch_geometric at all?"""
    import torch_geometric
    print(f"  torch_geometric version: {torch_geometric.__version__}")

def test_dense_to_sparse():
    """Does the built-in dense_to_sparse utility work?"""
    import torch
    from torch_geometric.utils import dense_to_sparse
    adj = torch.tensor([[0., 1.], [1., 0.]])
    ei, ev = dense_to_sparse(adj)
    assert ei.shape == (2, 2), f"Expected shape (2,2), got {ei.shape}"

def test_gcn_conv():
    """Can we run a single GCNConv forward pass?"""
    import torch
    from torch_geometric.nn import GCNConv
    conv = GCNConv(8, 8)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]])
    out = conv(x, edge_index)
    assert out.shape == (4, 8), f"Expected shape (4,8), got {out.shape}"

def test_gat_conv():
    """Can we run a single GATConv forward pass?"""
    import torch
    from torch_geometric.nn import GATConv
    conv = GATConv(8, 4, heads=2, concat=True)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]])
    out = conv(x, edge_index)
    assert out.shape == (4, 8), f"Expected shape (4,8), got {out.shape}"

def test_gin_conv():
    """Can we run a single GINConv forward pass?"""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GINConv
    mlp = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    conv = GINConv(mlp)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]])
    out = conv(x, edge_index)
    assert out.shape == (4, 8), f"Expected shape (4,8), got {out.shape}"

def test_sage_conv():
    """Can we run a single SAGEConv forward pass?"""
    import torch
    from torch_geometric.nn import SAGEConv
    conv = SAGEConv(8, 8)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]])
    out = conv(x, edge_index)
    assert out.shape == (4, 8), f"Expected shape (4,8), got {out.shape}"

def test_cuda_conv():
    """Can we run a GCNConv on GPU (if available)?"""
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return
    from torch_geometric.nn import GCNConv
    dev = torch.device("cuda:0")
    conv = GCNConv(8, 8).to(dev)
    x = torch.randn(4, 8, device=dev)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]], device=dev)
    out = conv(x, edge_index)
    assert out.shape == (4, 8), f"Expected shape (4,8), got {out.shape}"
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    tests = [
        test_import,
        test_dense_to_sparse,
        test_gcn_conv,
        test_gat_conv,
        test_gin_conv,
        test_sage_conv,
        test_cuda_conv,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"PASS  {name}")
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
