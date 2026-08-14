"""The single-network (FGNN) rows must not see the planted permutation.

Synthetic pairs carry a seed channel (``B[1,i,i] = i/n``) that the chain fills with
the previous link's prediction. On the un-chained first pass it has to be zeroed,
because ``all_perm`` permutes only graph A — which makes that channel *equal to* the
planted permutation. ``masking_noseed`` does the zeroing, but only from
``Base_Generator.__getitem__``, so handing a loader the dataset's raw ``.data`` list
silently leaks the ground truth into the model's input.

That is exactly what repro/reproduce_results.py did, inflating every synthetic
``FGNN Proj``/``FGNN FAQ`` cell (e.g. sparse ER d=4 at noise 0.2: 0.38/0.92 leaked
vs 0.12/0.34 masked). These tests pin the invariant.
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from loaders import get_data
from repro.reproduce_results import unseeded_loader

N_VERTICES = 24
N_PAIRS = 3


def _testset(tmp_path):
    cfg = OmegaConf.create({
        "type": "synthetic", "n_vertices": N_VERTICES, "generative_model": "ErdosRenyi",
        "noise_model": "ErdosRenyi", "edge_density": 0.2, "noise": 0.1, "seed": 0,
        "test": {"num_examples": N_PAIRS},
    })
    return get_data(cfg, str(tmp_path), saving=False, split="test")


def test_seed_channel_would_leak_the_planted_permutation(tmp_path):
    """Why the masking is load-bearing: unmasked, the seed channel *is* the answer."""
    gen = _testset(tmp_path)
    for graph_a, _graph_b, planted in gen.data:
        encoded = (torch.diagonal(graph_a[1]) * N_VERTICES).round().long().numpy()
        assert (encoded == np.argmax(planted.numpy(), 1)).all()


def test_unseeded_loader_zeroes_the_seed_channel(tmp_path):
    """The loader the FGNN row is built from must hide it again."""
    gen = _testset(tmp_path)
    seen = 0
    for data1, data2, _target in unseeded_loader(gen):
        assert torch.all(data1["input"][:, 1] == 0)
        assert torch.all(data2["input"][:, 1] == 0)
        assert data1["input"][:, 0].abs().sum() > 0  # adjacency survived
        seen += data1["input"].shape[0]
    assert seen == N_PAIRS


def test_unseeded_loader_rejects_a_raw_list(tmp_path):
    """Passing .data is the bug; it must not be silently accepted."""
    gen = _testset(tmp_path)
    with pytest.raises(TypeError, match="masking_noseed"):
        unseeded_loader(gen.data)
