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
