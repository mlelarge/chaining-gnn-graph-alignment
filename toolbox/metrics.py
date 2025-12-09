import torch
import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from scipy.special import log_softmax


def accuracy_max(weights, labels=None, aggregate_score=True):
    """
    weights should be (bs,n,n) and labels (bs,n,n) numpy arrays
    """
    acc = 0
    all_acc = []
    total_n_vertices = 0
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i].cpu().detach().numpy()
            # if label.ndim == 2:
            #    label = np.argmax(label, 1)
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


def perm2mat(p):
    n = np.max(p.shape)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, p[i]] = 1
    return P


def get_perm(ind_pair):
    ind0, ind1 = ind_pair
    perm = np.zeros(len(ind0))
    for i, j in enumerate(ind0):
        perm[j] = ind1[i]
    perm = np.int32(perm)
    return perm


def get_ranking(weight, graph1, graph2, use_faq=False):
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
    """
    weights should be (bs,n,n) and labels (bs,n) numpy arrays
    """
    total_n_vertices = 0
    acc = 0
    all_acc = []
    # weights = torch.log_softmax(rawscores,-1)
    weights = log_softmax(rawscores, axis=-1)
    for i, weight in enumerate(weights):
        if labels is not None:
            label = labels[i].cpu().detach().numpy()
            if label.ndim == 2:
                label = np.argmax(label, 1)
        else:
            label = np.arange(len(weight))
        cost = -weight  # .cpu().detach().numpy()
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
    all_qap = []
    all_d = []
    all_planted = []
    all_acc = []
    all_accd = []
    all_nit = []
    all_accmax = []
    # model = model.half()
    for batch in loader:
        (data1, data2, target) = batch
        data1["input"] = data1["input"].to(device)
        data2["input"] = data2["input"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            rawscores = model(data1, data2)
        weights = torch.log_softmax(rawscores, -1)
        g1 = data1["input"][:, 0, :, :].cpu().detach().numpy()
        g2 = data2["input"][:, 0, :, :].cpu().detach().numpy()
        planted = target.cpu().detach().numpy()

        n = len(planted[0])

        for i, weight in enumerate(weights):
            if planted[i].ndim == 2:
                pl = np.argmax(planted[i], 1)
            cost = -weight.cpu().detach().numpy()
            col_max = np.argmax(-cost, 1)
            row_ind, col_ind = linear_sum_assignment(cost)
            Pp = perm2mat(col_ind)
            res_qap = quadratic_assignment(
                g1[i], -g2[i], method="faq", options={"P0": Pp}
            )
            all_planted.append((g1[i] * g2[i][pl, :][:, pl]).sum() / 2)
            all_qap.append(
                (g1[i] * g2[i][res_qap["col_ind"], :][:, res_qap["col_ind"]]).sum() / 2
            )
            all_d.append((g1[i] * g2[i][col_ind, :][:, col_ind]).sum() / 2)
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
