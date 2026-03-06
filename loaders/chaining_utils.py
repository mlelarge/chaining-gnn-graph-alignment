"""Inference and chaining utilities for iterative graph alignment."""

import torch
import numpy as np
import copy
from toolbox.metrics import get_ranking, get_perm
from loaders.representations import adjacency_matrix_to_tensor_representation_ind


def all_ind(
    loader,
    model,
    device,
    compute_nce=False,
    random_order=False,
    use_faq=False,
    compute_faq=False,
    verbose=False,
    size_seed=0,
):
    ind_data = []
    model = model.to(device)
    all_nce = []
    all_faq = []
    all_acc = []
    with torch.no_grad():
        for batch in loader:
            data1, data2 = batch[0], batch[1]
            has_target = len(batch) == 3
            data1["input"] = data1["input"].to(device)
            data2["input"] = data2["input"].to(device)
            n_vertices = data1["input"].shape[-1]
            rawscores = model(data1, data2)
            rawscores = rawscores.to(torch.float32).cpu().detach()
            planted = batch[2].cpu().detach().numpy() if has_target else None
            weights = torch.log_softmax(rawscores, -1)
            g1 = copy.deepcopy(data1["input"][:, 0, :, :].cpu().detach().numpy())
            g2 = copy.deepcopy(data2["input"][:, 0, :, :].cpu().detach().numpy())
            for i, weight in enumerate(weights):
                ind1, col_ind = get_ranking(weight.numpy(), g1[i], g2[i], use_faq)
                pl = np.argmax(planted[i], 1) if has_target else None
                if random_order:
                    ind1 = np.random.permutation(len(ind1))
                if size_seed > 0 and pl is not None:
                    col_ind = np.concatenate((pl[:size_seed], col_ind[size_seed:]))
                ind2 = col_ind[ind1]
                ind_data.append((ind1, ind2))
                if compute_nce:
                    all_nce.append((g1[i] * g2[i][col_ind, :][:, col_ind]).sum() / 2)
                if compute_faq and not use_faq:  # only if use_faq is False
                    _, col_ind_faq = get_ranking(weight.numpy(), g1[i], g2[i], True)
                    nce_faq = (g1[i] * g2[i][col_ind_faq, :][:, col_ind_faq]).sum() / 2
                    nce_lap = (g1[i] * g2[i][col_ind, :][:, col_ind]).sum() / 2
                    all_faq.append(nce_faq)
                    if pl is not None:
                        acc = np.sum(pl == col_ind_faq) / n_vertices
                        all_acc.append(acc)
            del g1
            del g2
        if verbose and all_faq:
            print(
                f"NCE FAQ : {np.mean(all_faq)}, NCE LAP : {np.mean(all_nce)}, acc : {np.mean(all_acc)}"
            )
    if compute_nce:
        all_nce = np.array(all_nce)
        return ind_data, all_nce, np.array(all_faq) if compute_faq else None
    else:
        return ind_data, None


def make_data_from_ind(data, ind):
    return list(
        [adjacency_matrix_to_tensor_representation_ind(d, i) for d, i in zip(data, ind)]
    )


def make_data_from_ind_label(data, ind_pair):
    d1 = [d[0] for d in data]
    d2 = [d[1] for d in data]
    i1 = [i[0] for i in ind_pair]
    i2 = [i[1] for i in ind_pair]
    newd1, newd2 = make_data_from_ind(d1, i1), make_data_from_ind(d2, i2)
    if len(data[0]) == 3:
        label = [d[2] for d in data]
        return list(zip(newd1, newd2, label))
    return list(zip(newd1, newd2))
