"""Tests for the FAQ-initialization comparison (toolbox/baselines.evaluate_faq_inits).

The paper's story is FAQ(D_cx) vs FAQ(J); this checks the helper that computes both
(plus the Max-nce ceiling) behaves sensibly. Run with pytest or standalone.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolbox.baselines import evaluate_faq_inits


def _rand_graph(n, density, rng):
    u = np.triu((rng.random((n, n)) < density).astype(np.float64), 1)
    return u + u.T


def test_evaluate_faq_inits_exact_isomorphism():
    rng = np.random.default_rng(0)
    n = 30
    A = _rand_graph(n, 0.3, rng)
    perm = rng.permutation(n)
    B = A[np.ix_(perm, perm)]
    edges = A.sum() / 2

    r = evaluate_faq_inits(A, B, perm)

    # D_cx is a convex relaxation: it reliably recovers an exact isomorphism,
    # and FAQ seeded from the true permutation is by construction optimal.
    assert r["acc_dcx"] == 1.0
    assert r["acc_proj"] == 1.0
    assert np.isclose(r["nce_dcx"], edges)
    assert np.isclose(r["nce_max"], edges)
    assert np.isclose(r["nce_planted"], edges)
    # J init is a local method — valid range, not guaranteed exact.
    assert 0.0 <= r["acc_j"] <= 1.0
    assert r["nce_j"] <= edges + 1e-9


def test_evaluate_faq_inits_noisy_pair():
    rng = np.random.default_rng(7)
    n = 30
    A = _rand_graph(n, 0.3, rng)
    perm = rng.permutation(n)
    B = A[np.ix_(perm, perm)].copy()
    # Flip a handful of edges to break the exact isomorphism (keep symmetric).
    for _ in range(15):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            B[a, b] = 1.0 - B[a, b]
            B[b, a] = B[a, b]

    r = evaluate_faq_inits(A, B, perm)

    for k in ("acc_dcx", "acc_j", "acc_proj"):
        assert 0.0 <= r[k] <= 1.0, k
    for k in ("nce_dcx", "nce_j", "nce_max", "nce_planted"):
        assert r[k] >= 0, k
    # FAQ seeded from the true permutation is at least as good as the planted overlap.
    assert r["nce_max"] >= r["nce_planted"] - 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
