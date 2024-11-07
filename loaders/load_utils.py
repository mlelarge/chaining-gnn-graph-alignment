import torch
import copy

def masking_noseed(x):
    n = x.size(-1)
    y = copy.deepcopy(x)
    y[1,:,:] = torch.zeros(n,n)
    return y