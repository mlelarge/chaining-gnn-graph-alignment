import torch
import copy

def masking_noseed(x):
    n = x.size(-1)
    #y = copy.deepcopy(x)
    x[1,:,:] = torch.zeros(n,n)
    #del x
    #return y
    pass