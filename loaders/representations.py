"""Tensor representations of adjacency matrices."""

import torch
import numpy as np


def adjacency_matrix_to_tensor_representation(W):
    """Create a tensor B[0,:,:] = W and B[1,i,i] = i/n"""
    degrees = W.sum(1)
    B = torch.zeros((2, len(W), len(W)))
    B[0, :, :] = W
    indices = np.arange(len(W))
    B[1, indices, indices] = torch.tensor(indices / len(W), dtype=torch.float)
    return B


def adjacency_matrix_to_tensor_representation_ind(W, ind=None):
    """Create a tensor B = W except on the second diag B[1,j,j] = i where j = ind[i]"""
    n = W.shape[-1]
    B = torch.zeros((2, n, n))
    B[:] = W[:]
    B[1, range(n), range(n)] = torch.zeros(n)
    if ind is not None:
        for i, j in enumerate(ind):
            B[1, j, j] = torch.tensor((i) / n, dtype=torch.float)
    return B
