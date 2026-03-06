import torch
import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from scipy.special import log_softmax

from toolbox.utils import perm2mat


def accuracy_max(weights, labels=None, aggregate_score=True):
    """Compute accuracy using argmax matching.

    Args:
        weights: (bs, n, n) tensor of match scores.
        labels: (bs, n, n) tensor of ground-truth assignments (or None for identity).
        aggregate_score: If True, return totals; if False, return per-sample list.

    Returns:
        If aggregate_score: (acc, total_n_vertices) tuple.
        Otherwise: list of per-sample accuracy floats.
    """
    acc = 0
    all_acc = []
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i].cpu().detach().numpy()
            label = np.argmax(label, -1)
        else:
            label = np.arange(len(weight))
        weight = weight.to(torch.float32).cpu().detach().numpy()
        preds = np.argmax(weight, 1)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc


def get_perm(ind_pair):
    """Convert linear assignment index pairs to a permutation array.

    Args:
        ind_pair: Tuple (ind0, ind1) from linear_sum_assignment.

    Returns:
        int32 permutation array where perm[ind0[i]] = ind1[i].
    """
    ind0, ind1 = ind_pair
    perm = np.zeros(len(ind0))
    for i, j in enumerate(ind0):
        perm[j] = ind1[i]
    perm = np.int32(perm)
    return perm


def get_ranking(weight, graph1, graph2, use_faq=False):
    """Solve linear assignment and rank nodes by edge overlap score.

    Args:
        weight: (n, n) cost matrix to maximize.
        graph1, graph2: (n, n) adjacency matrices.
        use_faq: If True, refine the assignment using FAQ (quadratic_assignment).

    Returns:
        (row_ordering, col_ind):
        - row_ordering: Node indices sorted by ascending edge overlap score.
        - col_ind: Optimal column assignment (permutation).
    """
    _, col_ind = linear_sum_assignment(weight, maximize=True)
    if use_faq:
        Pp = perm2mat(col_ind)
        res_qap = quadratic_assignment(
            graph1, -graph2, method="faq", options={"P0": Pp}
        )
        col_ind = res_qap["col_ind"]

    maxi = (graph1 * graph2[col_ind, :][:, col_ind]).sum(1)
    return np.argsort(maxi), col_ind


def accuracy_linear_assignment(rawscores, labels=None, aggregate_score=True):
    """Compute accuracy using linear assignment on log-softmax scores.

    .. deprecated::
        This function is currently unused and may be removed in a future version.

    Args:
        rawscores: (bs, n, n) raw score array.
        labels: (bs, n) or (bs, n, n) target labels (or None for identity).
        aggregate_score: If True, return totals; if False, return per-sample list.

    Returns:
        If aggregate_score: (acc, total_n_vertices) tuple.
        Otherwise: list of per-sample accuracy floats.
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    weights = log_softmax(rawscores, axis=-1)
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i].cpu().detach().numpy()
            if label.ndim == 2:
                label = np.argmax(label, 1)
        else:
            label = np.arange(len(weight))
        cost = -weight
        _, preds = linear_sum_assignment(cost)
        if aggregate_score:
            acc += np.sum(preds == label)
            total_n_vertices += len(weight)
        else:
            all_acc += [np.sum(preds == label) / len(weight)]

    if aggregate_score:
        return acc, total_n_vertices
    else:
        return all_acc


def all_qap_chain(loader, model, device, verbose=False):
    """Evaluate a trained model on graph alignment using linear assignment + FAQ.

    Runs the model on each batch, solves the assignment problem on the predicted
    scores, then refines with FAQ. Compares against the planted ground truth.

    Args:
        loader: Dataloader yielding (data1, data2, target) batches.
        model: Trained siamese network model.
        device: Torch device (cuda/cpu).
        verbose: If True, also return FAQ iteration counts.

    Returns:
        Without verbose: 6 numpy arrays
            (planted, qap, d, acc, accd, accmax).
        With verbose: 7 numpy arrays (above + nit).
    """
    all_qap = []
    all_d = []
    all_planted = []
    all_acc = []
    all_accd = []
    all_nit = []
    all_accmax = []
    for batch in loader:
        data1, data2 = batch[0], batch[1]
        has_target = len(batch) == 3
        data1["input"] = data1["input"].to(device)
        data2["input"] = data2["input"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            rawscores = model(data1, data2)
        weights = torch.log_softmax(rawscores, -1)
        g1 = data1["input"][:, 0, :, :].cpu().detach().numpy()
        g2 = data2["input"][:, 0, :, :].cpu().detach().numpy()
        if has_target:
            planted = batch[2].cpu().detach().numpy()
            n = len(planted[0])

        for i, weight in enumerate(weights):
            if has_target and planted[i].ndim == 2:
                pl = np.argmax(planted[i], 1)
            cost = -weight.cpu().detach().numpy()
            col_max = np.argmax(-cost, 1)
            row_ind, col_ind = linear_sum_assignment(cost)
            Pp = perm2mat(col_ind)
            res_qap = quadratic_assignment(
                g1[i], -g2[i], method="faq", options={"P0": Pp}
            )
            all_qap.append(
                (g1[i] * g2[i][res_qap["col_ind"], :][:, res_qap["col_ind"]]).sum() / 2
            )
            all_d.append((g1[i] * g2[i][col_ind, :][:, col_ind]).sum() / 2)
            if has_target:
                all_planted.append((g1[i] * g2[i][pl, :][:, pl]).sum() / 2)
                all_acc.append(np.sum(pl == res_qap["col_ind"]) / n)
                all_accd.append(np.sum(pl == col_ind) / n)
                all_accmax.append(np.sum(pl == col_max) / n)
            if verbose:
                all_nit.append(res_qap["nit"])
    if verbose:
        return (
            np.array(all_planted),
            np.array(all_qap),
            np.array(all_d),
            np.array(all_acc),
            np.array(all_accd),
            np.array(all_accmax),
            np.array(all_nit),
        )
    else:
        return (
            np.array(all_planted),
            np.array(all_qap),
            np.array(all_d),
            np.array(all_acc),
            np.array(all_accd),
            np.array(all_accmax),
        )
