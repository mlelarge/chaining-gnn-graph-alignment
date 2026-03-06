import torch
import numpy as np


def masking_noseed(x):
    """Zero out the positional encoding channel (channel 1) in-place."""
    n = x.size(-1)
    x[1, :, :] = torch.zeros(n, n)


def recursive_tolist(obj):
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return [recursive_tolist(item) for item in obj]
    elif isinstance(obj, list):
        return [recursive_tolist(item) for item in obj]
    else:
        return obj
