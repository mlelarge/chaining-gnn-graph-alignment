import torch
import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment

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
            if label.ndim == 2:
                label = np.argmax(label,1)
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
    P = np.zeros((n,n))
    for i in range(n):
        P[i, p[i]] = 1
    return P

def get_ranking(weight, graph1, graph2, use_faq = False):
    if use_faq:
        Pp = perm2mat(col_ind)
        res_qap = quadratic_assignment(graph1,-graph2,method='faq',options={'P0':Pp})
        col_ind = res_qap['col_ind']
    else:
        _, col_ind = linear_sum_assignment(weight, maximize=True)
    maxi = (graph1 * graph2[col_ind,:][:,col_ind]).sum(1)
    return np.argsort(maxi), col_ind