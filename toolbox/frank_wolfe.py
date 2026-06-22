"""Frank-Wolfe solver for the *convex* graph-matching relaxation (``D_cx``).

This module isolates one of the paper's central contributions: solving graph
alignment via the **convex** relaxation of the quadratic assignment distance,

    minimize   ||A P - P B||_F^2     over doubly-stochastic matrices P,

with a Frank-Wolfe (conditional-gradient) method, optionally with seeded
(pre-matched) nodes. We refer to this objective / solver as ``D_cx`` — the
*convex distance* relaxation.

Why this matters (``D_cx`` vs ``J``)
------------------------------------
The standard FAQ baseline (``scipy.optimize.quadratic_assignment(method="faq")``)
optimizes the *indefinite* QAP objective ``-trace(A P B^T P^T)`` and is, in
practice, initialized at the barycenter ``J = 11^T / n`` (the "flat" doubly
stochastic matrix) — we call that variant ``FAQ(J)``. The objective here is
instead the *convex* surrogate ``||A P - P B||_F^2``; optimizing it to
convergence and projecting onto a permutation gives the ``FAQ(D_cx)`` solution.
The gap between ``FAQ(D_cx)`` and ``FAQ(J)`` is precisely the empirical effect
the paper reports, so the solver is kept here as a first-class, documented,
independently testable component rather than buried in the baselines module.

Canonical settings
-------------------
The values used for the paper's results are the defaults below:

    max_iter = 1000     # cap on Frank-Wolfe iterations
    tol      = 5e-2     # stop once the objective ||AP-PB||_F^2 falls below this
    tol2     = 1e-4     # stop once the objective stops decreasing (stagnation)

They are exposed as keyword arguments so they can be documented and tuned, but
the defaults reproduce the published behavior exactly.

Reference
---------
Adapted from the MATLAB implementation accompanying *Fast Approximate Quadratic
Programming for Graph Matching* (Vogelstein et al.):
https://github.com/jovo/FastApproximateQAP/blob/master/code/SGM/relaxed_normAPPB_FW_seeds.m
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FrankWolfeResult:
    """Structured result of the D_cx Frank-Wolfe solver.

    Attributes:
        P: Relaxed doubly-stochastic solution (transposed, as returned by the solver).
        col_ind: Projected permutation (column assignment) from linear assignment.
        n_iter: Number of Frank-Wolfe iterations taken (None if not requested).
    """

    P: np.ndarray
    col_ind: np.ndarray
    n_iter: Optional[int] = None


def fro_norm(P, A, B):
    """Compute the squared Frobenius norm ``||A P - P B||_F^2`` (the D_cx objective)."""
    return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord="fro") ** 2


def indef_rel(P, A, B):
    """Compute the indefinite relaxation ``-trace(A^T P B^T P)``.

    .. deprecated::
        This function is currently unused and may be removed in a future version.
        Kept here alongside the convex objective for reference/comparison.
    """
    return -np.trace(np.transpose(A @ P) @ (P @ B))


def relaxed_normAPPB_FW_seeds(
    A, B, max_iter=1000, seeds=0, verbose=False, tol=5e-2, tol2=1e-4
):
    """Frank-Wolfe solver for the convex (D_cx) graph-matching relaxation, with seeds.

    Minimizes ``||A P - P B||_F^2`` over doubly-stochastic matrices using the
    Frank-Wolfe (conditional gradient) method, optionally fixing the first
    ``seeds`` nodes to a known correspondence.

    Args:
        A, B: (n, n) adjacency matrices of the two graphs.
        max_iter: Maximum number of Frank-Wolfe iterations (default 1000).
        seeds: Number of pre-matched (seeded) node correspondences.
        verbose: If True, return the iteration count as the third value.
        tol: Objective threshold — stop once ``||AP-PB||_F^2 < tol`` (default 5e-2).
        tol2: Stagnation threshold — stop once the objective change < tol2 (default 1e-4).

    Returns:
        (P, col_ind, s):
        - P: Relaxed doubly-stochastic solution (transposed).
        - col_ind: Projected permutation from linear assignment.
        - s: Iteration count (None if verbose=False).
    """
    AtA = np.dot(A.T, A)
    BBt = np.dot(B, B.T)
    p = A.shape[0]

    def f1(P):
        return np.linalg.norm(np.dot(A, P) - np.dot(P, B), ord="fro") ** 2

    P = np.ones((p, p)) / (p - seeds)
    P[:seeds, :seeds] = np.eye(seeds)

    f = f1(P)
    var = 1
    s = 0

    while not (np.abs(f) < tol) and (var > tol2) and (s < max_iter):
        fold = f

        grad = 2 * (
            np.dot(AtA, P)
            - np.dot(np.dot(A.T, P), B)
            - np.dot(np.dot(A, P), B.T)
            + np.dot(P, BBt)
        )

        grad[:seeds, :] = 0
        grad[:, :seeds] = 0

        row_ind, col_ind = linear_sum_assignment(grad[seeds:, seeds:])

        # Embed the sub-problem assignment back into the full n x n permutation,
        # keeping the seeded block fixed to the identity. For seeds == 0 this is
        # exactly perm2mat(col_ind) (the original behavior, bit-for-bit). The
        # explicit construction also fixes the seeded path: the previous
        # perm2mat(col_ind) produced an (n-seeds) x (n-seeds) matrix and crashed
        # whenever seeds > 0.
        Ps = np.zeros((p, p))
        Ps[:seeds, :seeds] = np.eye(seeds)
        Ps[seeds + row_ind, seeds + col_ind] = 1.0

        C = np.dot(A, P - Ps) + np.dot(Ps - P, B)
        D = np.dot(A, Ps) - np.dot(Ps, B)

        aq = np.trace(np.dot(C, C.T))
        bq = np.trace(np.dot(C, D.T) + np.dot(D, C.T))
        aopt = -bq / (2 * aq)

        Ps4 = aopt * P + (1 - aopt) * Ps

        f = f1(Ps4)
        P = Ps4

        var = np.abs(f - fold)
        s += 1

    _, col_ind = linear_sum_assignment(-P.T)

    if verbose:
        return P.T, col_ind, s
    else:
        return P.T, col_ind, None


def solve_dcx(A, B, max_iter=1000, seeds=0, tol=5e-2, tol2=1e-4):
    """Solve the convex (D_cx) graph-matching relaxation; return a structured result.

    Thin, documented public wrapper over :func:`relaxed_normAPPB_FW_seeds`. The
    numerics are identical to calling that function directly with ``verbose=True``.

    Args:
        A, B: (n, n) adjacency matrices.
        max_iter, seeds, tol, tol2: see :func:`relaxed_normAPPB_FW_seeds`.

    Returns:
        FrankWolfeResult with fields ``P``, ``col_ind`` and ``n_iter``.
    """
    P, col_ind, n_iter = relaxed_normAPPB_FW_seeds(
        A, B, max_iter=max_iter, seeds=seeds, verbose=True, tol=tol, tol2=tol2
    )
    return FrankWolfeResult(P=P, col_ind=col_ind, n_iter=n_iter)
