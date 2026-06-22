"""Tests for the D_cx Frank-Wolfe graph-matching solver (toolbox/frank_wolfe.py).

Covers:
  * backward-compatible re-export from toolbox.baselines,
  * equivalence of the solve_dcx() wrapper and the raw solver,
  * correctness: recovery of a known graph isomorphism (with and without seeds).

Run with pytest, or standalone:  python tests/test_frank_wolfe.py
"""

import os
import sys

import numpy as np

# Allow `python tests/test_frank_wolfe.py` (standalone) as well as pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolbox.frank_wolfe import (
    FrankWolfeResult,
    fro_norm,
    relaxed_normAPPB_FW_seeds,
    solve_dcx,
)


def _random_symmetric_graph(n, density, rng):
    """Return an (n, n) symmetric 0/1 adjacency with zero diagonal."""
    upper = (rng.random((n, n)) < density).astype(np.float64)
    upper = np.triu(upper, k=1)
    adj = upper + upper.T
    return adj


def _permuted_copy(adj, perm):
    """Relabel `adj` by `perm`: B[i, j] = adj[perm[i], perm[j]]."""
    return adj[np.ix_(perm, perm)]


def test_backward_compat_reexport():
    """`from toolbox.baselines import ...` must still resolve to the moved symbols."""
    from toolbox import baselines

    assert baselines.relaxed_normAPPB_FW_seeds is relaxed_normAPPB_FW_seeds
    assert baselines.fro_norm is fro_norm


def test_solve_dcx_matches_raw_solver():
    """The structured wrapper returns numerics identical to the raw solver."""
    rng = np.random.default_rng(0)
    A = _random_symmetric_graph(25, 0.3, rng)
    perm = rng.permutation(25)
    B = _permuted_copy(A, perm)

    P_raw, col_raw, n_iter_raw = relaxed_normAPPB_FW_seeds(A, B, verbose=True)
    res = solve_dcx(A, B)

    assert isinstance(res, FrankWolfeResult)
    assert np.array_equal(res.col_ind, col_raw)
    assert np.allclose(res.P, P_raw)
    assert res.n_iter == n_iter_raw


def test_recovers_isomorphism():
    """On an exact relabeling, the D_cx solver should recover a perfect alignment."""
    rng = np.random.default_rng(1)
    n = 30
    A = _random_symmetric_graph(n, 0.3, rng)
    perm = rng.permutation(n)
    B = _permuted_copy(A, perm)

    _, col, _ = relaxed_normAPPB_FW_seeds(A, B, verbose=True)

    # Aligning A by `col` must reproduce B exactly (edge overlap fully recovered).
    assert np.array_equal(A[np.ix_(col, col)], B)
    assert (B * A[np.ix_(col, col)]).sum() == B.sum()


def test_fro_norm_zero_at_truth():
    """The convex objective ||AP - PB||_F^2 is exactly zero at the recovered matching."""
    rng = np.random.default_rng(2)
    n = 20
    A = _random_symmetric_graph(n, 0.35, rng)
    perm = rng.permutation(n)
    B = _permuted_copy(A, perm)

    from toolbox.utils import perm2mat

    _, col, _ = relaxed_normAPPB_FW_seeds(A, B, verbose=True)
    # perm2mat(col)[i, col[i]] = 1 = Q; A[col][:,col] == B means Q A Q^T == B,
    # so P = Q^T satisfies A P == P B and the convex objective is exactly zero.
    Q = perm2mat(col)
    assert fro_norm(Q.T, A, B) < 1e-9


def test_seeded_recovery():
    """With the first nodes seeded as identity, recovery still holds."""
    rng = np.random.default_rng(3)
    n = 24
    seeds = 4
    A = _random_symmetric_graph(n, 0.3, rng)
    # A permutation that fixes the first `seeds` nodes (consistent with identity seeds).
    perm = np.concatenate([np.arange(seeds), seeds + rng.permutation(n - seeds)])
    B = _permuted_copy(A, perm)

    _, col, _ = relaxed_normAPPB_FW_seeds(A, B, seeds=seeds, verbose=True)
    assert np.array_equal(A[np.ix_(col, col)], B)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
