"""Tests for the BAPG Gromov-Wasserstein baseline (toolbox/bapg).

The exact-isomorphism test is the important one: it mechanically locks the
orientation of the transport-plan decode (T.T, not T) against the repo's
planted-permutation convention. Run with pytest or standalone.
"""

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolbox.bapg import evaluate_bapg, solve_bapg


def _rand_graph(n, density, rng):
    u = np.triu((rng.random((n, n)) < density).astype(np.float64), 1)
    return u + u.T


def test_evaluate_bapg_exact_isomorphism():
    rng = np.random.default_rng(0)
    n = 60
    A = _rand_graph(n, 0.2, rng)
    perm = rng.permutation(n)
    B = A[np.ix_(perm, perm)]
    edges = A.sum() / 2

    r = evaluate_bapg(A, B, perm)

    # The paper reports ~100% on noise-free pairs; anything below exact
    # recovery here means the decode orientation is wrong.
    assert r["acc_bapg"] == 1.0
    assert np.isclose(r["nce_bapg"], edges)


def test_evaluate_bapg_noisy_pair():
    rng = np.random.default_rng(7)
    n = 60
    A = _rand_graph(n, 0.2, rng)
    perm = rng.permutation(n)
    B = A[np.ix_(perm, perm)].copy()
    # Flip a handful of edges to break the exact isomorphism (keep symmetric).
    for _ in range(30):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            B[a, b] = 1.0 - B[a, b]
            B[b, a] = B[a, b]

    res = solve_bapg(A, B)
    # The plan may violate the marginals (that's the relaxation), but the
    # Hungarian decode must still be a bijection and the scores finite.
    assert np.array_equal(np.sort(res.col_ind), np.arange(n))

    r = evaluate_bapg(A, B, perm)
    assert 0.0 <= r["acc_bapg"] <= 1.0
    assert 0 <= r["nce_bapg"] <= A.sum() / 2 + 1e-9


def test_evaluate_bapg_regular_graph_smoke():
    # d-regular graphs are the degenerate regime for relaxation-based matching
    # (uniform degrees, flat initialization); BAPG only has to run to
    # completion and return valid finite scores, not to succeed.
    A = nx.to_numpy_array(nx.random_regular_graph(6, 40, seed=3))
    rng = np.random.default_rng(3)
    perm = rng.permutation(40)
    B = A[np.ix_(perm, perm)]

    r = evaluate_bapg(A, B, perm)
    assert np.isfinite(r["acc_bapg"]) and np.isfinite(r["nce_bapg"])
    assert 0.0 <= r["acc_bapg"] <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
