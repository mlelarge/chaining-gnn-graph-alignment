import torch
import copy
import numpy as np

def masking_noseed(x):
    n = x.size(-1)
    #y = copy.deepcopy(x)
    x[1,:,:] = torch.zeros(n,n)
    #del x
    #return y
    pass

def recursive_tolist(obj):
    """
    Recursively convert numpy arrays to lists
    """
    if isinstance(obj, np.ndarray):
        return [recursive_tolist(item) for item in obj]
    elif isinstance(obj, list):
        return [recursive_tolist(item) for item in obj]
    else:
        return obj