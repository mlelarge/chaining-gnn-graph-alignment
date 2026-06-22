import numpy as np
from scipy.optimize import quadratic_assignment

# The D_cx Frank-Wolfe solver lives in toolbox/frank_wolfe.py. It is re-exported
# here for backward compatibility (existing code does
# `from toolbox.baselines import relaxed_normAPPB_FW_seeds`).
from toolbox.frank_wolfe import (  # noqa: F401
    fro_norm,
    indef_rel,
    relaxed_normAPPB_FW_seeds,
)
from toolbox.utils import perm2mat


def evaluate_faq_inits(g1, g2, planted_perm, maxiter_faq=30):
    """Compare FAQ initializations on one graph pair — the paper's D_cx-vs-J story.

    Runs scipy's FAQ (`quadratic_assignment(method="faq")`) from three different
    starting points and reports accuracy vs the planted permutation and the
    edge-overlap ("common edges", nce) of each solution:

    - **D_cx**: initialize FAQ at the convex Frank-Wolfe solution
      (`toolbox.frank_wolfe.relaxed_normAPPB_FW_seeds`) — the paper's method.
    - **J**: initialize FAQ at the barycenter ``J`` (scipy's default) — the baseline.
    - **max**: initialize FAQ at the true permutation — the achievable edge-overlap
      ceiling ("Max-nce").

    Also returns the raw D_cx *projection* accuracy (the permutation read off the
    Frank-Wolfe relaxation before FAQ refinement).

    Args:
        g1, g2: (n, n) adjacency matrices of the two graphs.
        planted_perm: (n,) ground-truth permutation (argmax of the planted target).
        maxiter_faq: FAQ refinement iteration cap for the D_cx initialization.

    Returns:
        dict with keys acc_dcx, acc_j, acc_proj, nce_dcx, nce_j, nce_max, nce_planted.
    """
    pl = planted_perm
    n = len(pl)

    def overlap(col):
        return (g2 * g1[col, :][:, col]).sum() / 2

    # D_cx: Frank-Wolfe convex solution as the FAQ init.
    P, col_proj, _ = relaxed_normAPPB_FW_seeds(g1, g2)
    col_dcx = quadratic_assignment(
        g2, -g1, method="faq", options={"P0": P, "maxiter": maxiter_faq}
    )["col_ind"]
    # J: barycenter init (scipy default).
    col_j = quadratic_assignment(g2, -g1, method="faq")["col_ind"]
    # Max-nce: FAQ seeded from the true permutation.
    col_max = quadratic_assignment(
        g2, -g1, method="faq", options={"P0": perm2mat(pl)}
    )["col_ind"]

    return {
        "acc_dcx": np.sum(pl == col_dcx) / n,
        "acc_j": np.sum(pl == col_j) / n,
        "acc_proj": np.sum(pl == col_proj) / n,
        "nce_dcx": overlap(col_dcx),
        "nce_j": overlap(col_j),
        "nce_max": overlap(col_max),
        "nce_planted": overlap(pl),
    }


def baseline(loader):
    """Compute baseline QAP metrics over a dataloader.

    For each sample, evaluates the identity mapping, the planted (ground-truth)
    solution, and an unsupervised FAQ solution.

    Returns:
        Tuple of (all_b, all_u, all_acc, all_p) as numpy arrays:
        - all_b: edge overlap under identity mapping
        - all_u: edge overlap under FAQ solution
        - all_acc: accuracy of FAQ vs planted
        - all_p: edge overlap under planted solution
    """
    all_b = []
    all_u = []
    all_acc = []
    all_p = []
    for batch in loader:
        (data1, data2, target) = batch
        g1 = data1["input"][:, 0, :, :].cpu().detach().numpy()
        g2 = data2["input"][:, 0, :, :].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()
        n = len(planted[0])
        bs = planted.shape[0]
        for i in range(bs):
            all_b.append((g1[i] * g2[i]).sum() / 2)
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i], 1)
            all_p.append((g1[i] * g2[i][pl, :][:, pl]).sum() / 2)
            Pp = perm2mat(pl)
            res_qap = quadratic_assignment(
                g1[i], -g2[i], method="faq", options={"P0": Pp}
            )
            all_u.append(
                (g1[i] * g2[i][res_qap["col_ind"], :][:, res_qap["col_ind"]]).sum() / 2
            )
            all_acc.append(np.sum(pl == res_qap["col_ind"]) / n)
    return np.array(all_b), np.array(all_u), np.array(all_acc), np.array(all_p)


def all_qap_scipy(loader, max_iter=1000, maxiter_faq=30, seeds=0, verbose=False):
    """Evaluate graph alignment using Frank-Wolfe + FAQ on a dataloader.

    Computes multiple metrics comparing the Frank-Wolfe relaxation, FAQ
    refinement, and planted (ground-truth) solutions.

    Args:
        loader: Dataloader yielding (data1, data2, target) batches.
        max_iter: Max iterations for Frank-Wolfe.
        maxiter_faq: Max iterations for scipy FAQ refinement.
        seeds: Number of seeded correspondences.
        verbose: If True, also return iteration counts.

    Returns:
        Without verbose: 9 numpy arrays
            (planted, qap, d, acc, accd, fd, fproj, fqap, fplanted).
        With verbose: 11 numpy arrays (above + conv_nit, nit).
    """
    all_qap = []
    all_d = []
    all_planted = []
    all_acc = []
    all_accd = []
    all_fd = []
    all_fproj = []
    all_fqap = []
    all_fplanted = []
    all_conv_nit = []
    all_nit = []
    for batch in loader:
        (data1, data2, target) = batch
        g1 = data1["input"][:, 0, :, :].cpu().detach().numpy()
        g2 = data2["input"][:, 0, :, :].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()

        n = len(planted[0])
        bs = planted.shape[0]

        for i in range(bs):
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i], 0)
            P, col, s = relaxed_normAPPB_FW_seeds(
                g1[i], g2[i], max_iter=max_iter, seeds=seeds, verbose=verbose
            )
            if verbose:
                all_conv_nit.append(s)
            Pp = perm2mat(col)
            all_fd.append(fro_norm(P.T, g1[i], g2[i]))
            all_fproj.append(fro_norm(Pp.T, g1[i], g2[i]))
            res_qap = quadratic_assignment(
                g2[i], -g1[i], method="faq", options={"P0": P, "maxiter": maxiter_faq}
            )
            P_qap = perm2mat(res_qap["col_ind"])
            all_fqap.append(fro_norm(P_qap.T, g1[i], g2[i]))
            P_planted = perm2mat(pl)
            all_fplanted.append(fro_norm(P_planted.T, g1[i], g2[i]))

            all_planted.append((g2[i] * g1[i][pl, :][:, pl]).sum() / 2)
            all_qap.append(
                (g2[i] * g1[i][res_qap["col_ind"], :][:, res_qap["col_ind"]]).sum() / 2
            )
            all_d.append((g2[i] * g1[i][col, :][:, col]).sum() / 2)
            all_acc.append(np.sum(pl == res_qap["col_ind"]) / n)
            all_accd.append(np.sum(pl == col) / n)
            if verbose:
                all_nit.append(res_qap["nit"])
    if verbose:
        return (
            np.array(all_planted),
            np.array(all_qap),
            np.array(all_d),
            np.array(all_acc),
            np.array(all_accd),
            np.array(all_fd),
            np.array(all_fproj),
            np.array(all_fqap),
            np.array(all_fplanted),
            np.array(all_conv_nit),
            np.array(all_nit),
        )
    else:
        return (
            np.array(all_planted),
            np.array(all_qap),
            np.array(all_d),
            np.array(all_acc),
            np.array(all_accd),
            np.array(all_fd),
            np.array(all_fproj),
            np.array(all_fqap),
            np.array(all_fplanted),
        )
