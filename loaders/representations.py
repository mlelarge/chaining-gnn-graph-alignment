"""Tensor representations of adjacency matrices."""

import torch
import numpy as np


def adjacency_matrix_to_tensor_representation(W):
    """Create a tensor B[0,:,:] = W and B[1,i,i] = i/n"""
    n = len(W)
    B = W.new_zeros((2, n, n))
    B[0, :, :] = W
    indices = torch.arange(n, device=W.device, dtype=torch.float)
    B[1, torch.arange(n, device=W.device), torch.arange(n, device=W.device)] = indices / n
    return B


def adjacency_matrix_to_tensor_representation_ind(W, ind=None):
    """Create a tensor B = W except on the second diag B[1,j,j] = i where j = ind[i]"""
    n = W.shape[-1]
    B = W.new_zeros((2, n, n))
    B[:] = W[:]
    B[1, range(n), range(n)] = W.new_zeros(n)
    if ind is not None:
        for i, j in enumerate(ind):
            B[1, j, j] = W.new_tensor(i / n)
    return B
