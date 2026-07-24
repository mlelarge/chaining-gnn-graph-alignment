"""Tests for the random_order ablation flag (loaders.chaining_utils.all_ind).

random_order must change only the inter-link *ordering* (which node gets which
rank value), never the underlying matching that is transported between graphs.
Run with pytest or standalone.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import loaders.chaining_utils as cu


class _StubModel:
    """Returns a fixed score matrix so the LAP matching is deterministic."""

    def __init__(self, scores):
        self._scores = scores  # (b, n, n)

    def to(self, device):
        return self

    def __call__(self, data1, data2):
        return self._scores


def _recover_matching(ind1, ind2, n):
    # ind2[i] = col_ind[ind1[i]]  =>  col_ind[ind1] = ind2
    col = np.empty(n, dtype=int)
    col[ind1] = ind2
    return col


def _one_batch(n, seed):
    rng = np.random.default_rng(seed)
    a = np.triu((rng.random((n, n)) < 0.3).astype(np.float32), 1)
    a = a + a.T
    g1 = torch.tensor(a)[None, None]  # (1,1,n,n)
    g1 = torch.cat([g1, torch.zeros_like(g1)], dim=1)  # (1,2,n,n)
    b = np.triu((rng.random((n, n)) < 0.3).astype(np.float32), 1)
    b = b + b.T
    g2 = torch.tensor(b)[None, None]
    g2 = torch.cat([g2, torch.zeros_like(g2)], dim=1)
    data1 = {"input": g1}
    data2 = {"input": g2}
    scores = torch.tensor(rng.random((1, n, n)), dtype=torch.float32)
    return [(data1, data2)], _StubModel(scores)


def test_random_order_preserves_matching_changes_order():
    n = 40
    loader, model = _one_batch(n, seed=1)
    r_score = cu.all_ind(loader, model, "cpu", random_order=False)
    loader, model = _one_batch(n, seed=1)  # identical inputs
    np.random.seed(0)  # make the random permutation deterministic + non-identity
    r_rand = cu.all_ind(loader, model, "cpu", random_order=True)

    ind1_s, ind2_s = r_score.indices[0]
    ind1_r, ind2_r = r_rand.indices[0]

    # Ordering is a valid permutation in both cases...
    assert np.array_equal(np.sort(ind1_s), np.arange(n))
    assert np.array_equal(np.sort(ind1_r), np.arange(n))
    # ...but the random one differs from the score-sorted one.
    assert not np.array_equal(ind1_s, ind1_r)
    # The transported matching is IDENTICAL — only the ordering changed.
    col_s = _recover_matching(ind1_s, ind2_s, n)
    col_r = _recover_matching(ind1_r, ind2_r, n)
    assert np.array_equal(col_s, col_r)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
