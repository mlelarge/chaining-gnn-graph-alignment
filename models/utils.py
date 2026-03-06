# Adapted from https://github.com/davidcpage/cifar10-fast/blob/master/core.py
import torch
import torch.nn as nn
from collections import defaultdict

#####################
## dict utils
#####################

def union(*dicts: dict) -> dict:
    return {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)

def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k,v in nested_dict.items()}

def group_by_key(items):
    res = defaultdict(list)
    for k, v in items:
        res[k].append(v)
    return res

#####################
## graph building
#####################
sep = '/'

def split(path):
    i = path.rfind(sep) + 1
    return path[:i].rstrip(sep), path[i:]

def normpath(path):
    # Simplified path normalization for the '/' separated graph node paths.
    # Not a replacement for os.path.normpath — this operates on logical graph
    # paths (e.g. 'block/mlp/../in' -> 'block/in'), not filesystem paths.
    parts = []
    for p in path.split(sep):
        if p == '..': parts.pop()
        elif p.startswith(sep): parts = [p]
        else: parts.append(p)
    return sep.join(parts)

def has_inputs(node) -> bool:
    return type(node) is tuple

def pipeline(net):
    return [(sep.join(path), (node if has_inputs(node) else (node, [-1]))) for (path, node) in path_iter(net)]

def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: normpath(sep.join((path, '..', rel_path))) if isinstance(rel_path, str) else flattened[idx+rel_path][0]
    return {path: (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]]) for idx, (path, node) in enumerate(flattened)}

class Network(nn.Module):
    """Declarative computation graph built from nested dictionaries.

    Nodes are defined as nested dicts where keys become '/'-separated paths.
    Each node is either a bare nn.Module (implicitly takes previous node's
    output) or a tuple (module, [input_paths]) where input_paths are relative
    string references (e.g. '../in') or integer offsets (e.g. -1 for previous).

    Example::

        net = {
            'in': Identity(),
            'block': {
                'mlp': (MlpBlock(...), ['../in']),
                'add': (Add(), ['../in', 'mlp']),
            },
        }
        model = Network(net)
        out = model({'input': x})
    """

    def __init__(self, net):
        super().__init__()
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items():
            setattr(self, path.replace('/', '_'), val)

    def nodes(self):
        return (node for node, _ in self.graph.values())

    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (node, ins) in self.graph.items():
            # Only compute nodes that are not supplied as inputs.
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])
        return outputs

    def half(self):
        # BatchNorm2d is excluded because it requires float32 for numerically
        # stable running mean/variance statistics.
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self
