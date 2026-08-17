"""Tests for get_ranking's rank_key option (toolbox/metrics).

"raw" must reproduce the historical behavior bit-for-bit (it is the default and
drives the released chains); "degree_normalized" is the ablation key. Run with
pytest or standalone.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolbox.metrics import get_ranking


def _pair():
    # Node 0: degree 4, 2 preserved edges (m=2, m/d=0.5)
    # Node 5: degree 1, 1 preserved edge   (m=1, m/d=1.0)
    # Raw ranks 0 above 5; degree-normalized ranks 5 above 0.
    n = 6
    g1 = np.zeros((n, n))
    for j in (1, 2, 3, 4):
        g1[0, j] = g1[j, 0] = 1.0
    g1[5, 1] = g1[1, 5] = 1.0
    g2 = g1.copy()
    g2[0, 3] = g2[3, 0] = 0.0  # break two of node 0's edges
    g2[0, 4] = g2[4, 0] = 0.0
    weight = np.eye(n)  # identity assignment
    return g1, g2, weight


def test_raw_default_unchanged():
    g1, g2, w = _pair()
    ind_default, col_default = get_ranking(w, g1, g2)
    ind_raw, col_raw = get_ranking(w, g1, g2, rank_key="raw")
    assert np.array_equal(ind_default, ind_raw)
    assert np.array_equal(col_default, col_raw)
    # raw key: node 0 (m=2) must rank above node 5 (m=1)
    assert list(ind_raw).index(0) > list(ind_raw).index(5)


def test_degree_normalized_changes_order():
    g1, g2, w = _pair()
    ind, col = get_ranking(w, g1, g2, rank_key="degree_normalized")
    assert np.array_equal(col, np.arange(6))
    # normalized key: node 5 (m/d=1.0) must rank above node 0 (m/d=0.5)
    assert list(ind).index(5) > list(ind).index(0)


def test_unknown_rank_key_raises():
    g1, g2, w = _pair()
    with pytest.raises(ValueError):
        get_ranking(w, g1, g2, rank_key="typo")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {exc}")
