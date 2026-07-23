"""BAPG solver for the Gromov-Wasserstein graph-matching relaxation.

This module wraps POT's ``ot.gromov.BAPG_gromov_wasserstein`` — the Bregman
Alternating Projected Gradient method of

    Li, Tang, Kong, Liu, Li, So, Blanchet, "A Convergent Single-Loop Algorithm
    for Relaxation of Gromov-Wasserstein in Graph Data", ICLR 2023
    (arXiv:2303.06595)

— as a graph-alignment baseline comparable, per pair, with the D_cx / FAQ
baselines (``toolbox.baselines``) and the chained GNN.

BAPG solves a *relaxation* of the GW problem that decouples the two marginal
constraints, enforcing only one per half-step via a KL (Bregman) projection.
Consequences to keep in mind:

- the returned transport plan may **violate the marginal constraints** (bounded
  by ``tau/epsilon`` in the paper's notation) — this is by design, not a bug;
- the loss reported by POT is not a true GW cost (it may be negative) and is
  not used here; only the plan is.

``epsilon`` is the paper's Bregman **step-size** ``rho``, *not* an entropic
regularizer: too small a value overflows the multiplicative update
(``exp(-grad/epsilon)``) and yields a non-finite plan.

Canonical settings
-------------------
The defaults below follow the paper's graph-alignment protocol (their Sec. 5:
raw 0/1 adjacency inputs, uniform marginals, rho = 0.1 with no tuning,
2000 iterations):

    epsilon  = 0.1      # Bregman step size (paper's rho)
    max_iter = 2000     # iteration cap
    tol      = 1e-6     # stop once the change in the plan falls below this

Note on ``tol``: the paper stops on the *relative* plan change; POT's ``tol``
is the **absolute** Frobenius norm of the plan change, checked every 10
iterations. At ``epsilon = 0.1`` the updates are large enough that this only
halts at genuine fixed points, so the distinction is harmless here — but do
not shrink ``epsilon`` and ``tol`` together expecting the paper's rule.

The paper extracts correspondences by row-wise argmax of the plan; here the
plan is instead projected to a *bijection* with the Hungarian algorithm
(``linear_sum_assignment``), matching how every other soft solution in this
repo is decoded (see ``toolbox/frank_wolfe.py``).
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


@dataclass
class BAPGResult:
    """Structured result of the BAPG Gromov-Wasserstein solver.

    Attributes:
        T: Transport plan (may violate the marginal constraints — by design).
        col_ind: Projected permutation (column assignment) from linear assignment.
        n_checks: Number of convergence checks POT recorded — one per 10
            iterations, so ~n_checks*10 iterations ran (None if unavailable).
    """

    T: np.ndarray
    col_ind: np.ndarray
    n_checks: Optional[int] = None


def solve_bapg(A, B, epsilon=0.1, max_iter=2000, tol=1e-6, loss_fun="square_loss"):
    """Run BAPG Gromov-Wasserstein on two adjacency matrices.

    Args:
        A, B: (n, n) symmetric adjacency matrices (used raw, uniform marginals).
        epsilon: Bregman step size (the paper's rho). Larger is safer but slower.
        max_iter: Iteration cap.
        tol: Stop once the change in the plan falls below this.
        loss_fun: POT loss ("square_loss" or "kl_loss").

    Returns:
        BAPGResult with the plan, its Hungarian projection (comparable to the
        planted permutation, same orientation as the D_cx decode), and the
        iteration count.

    Raises:
        FloatingPointError: if the plan is non-finite (epsilon too small).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    T, log = ot.gromov.BAPG_gromov_wasserstein(
        A, B, loss_fun=loss_fun, epsilon=epsilon, max_iter=max_iter, tol=tol, log=True
    )
    if not np.all(np.isfinite(T)):
        raise FloatingPointError(
            f"BAPG produced a non-finite transport plan (epsilon={epsilon} too small)"
        )
    # Same decode as the D_cx solution (frank_wolfe.py): transpose, then LAP.
    _, col_ind = linear_sum_assignment(-T.T)
    err = log.get("err")
    return BAPGResult(T=T, col_ind=col_ind, n_checks=len(err) if err is not None else None)


def evaluate_bapg(g1, g2, planted_perm, epsilon=0.1, max_iter=2000, tol=1e-6):
    """Evaluate BAPG-GW on one graph pair — accuracy and edge overlap.

    Mirrors ``toolbox.baselines.evaluate_faq_inits``: same input contract
    (``g2[i, j] ~ g1[pl[i], pl[j]]``), same metric definitions.

    A non-finite plan is retried once with ``10 * epsilon``; if that also
    fails, the identity permutation is scored so that grid runs always produce
    a complete record — the returned ``bapg_failed`` flag marks such pairs
    (callers should count and report them; the accompanying warning is
    deduplicated by Python after the first occurrence).

    Args:
        g1, g2: (n, n) adjacency matrices of the two graphs.
        planted_perm: (n,) ground-truth permutation (argmax of the planted target).
        epsilon, max_iter, tol: solver settings (see ``solve_bapg``).

    Returns:
        dict with keys acc_bapg, nce_bapg, bapg_failed (1.0 when the scores
        are the identity-permutation fallback, not a BAPG solution).
    """
    pl = planted_perm
    n = len(pl)
    failed = False
    try:
        col = solve_bapg(g1, g2, epsilon=epsilon, max_iter=max_iter, tol=tol).col_ind
    except FloatingPointError:
        try:
            warnings.warn(f"BAPG plan non-finite at epsilon={epsilon}; retrying with {10 * epsilon}")
            col = solve_bapg(g1, g2, epsilon=10 * epsilon, max_iter=max_iter, tol=tol).col_ind
        except FloatingPointError:
            warnings.warn("BAPG failed twice (non-finite plan); scoring the identity permutation")
            col = np.arange(n)
            failed = True

    return {
        "acc_bapg": np.sum(pl == col) / n,
        "nce_bapg": (g2 * g1[col, :][:, col]).sum() / 2,
        "bapg_failed": float(failed),
    }
