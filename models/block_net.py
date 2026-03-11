from models.layers import *

def block_emb(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp3': MlpBlock_Real(in_features, out_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices)
    }

def init_emb(in_features, out_features):
    return {
        'in': Identity(),
        'mlp3': Conv_norm(in_features, out_features)
    }


def node_emb(in_features, out_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'diag': (Diag(), ['in']),
        'mlp_node': (MlpBlock_Node(in_features, out_features, depth_of_mlp, constant_n_vertices=constant_n_vertices))
    }

def node_pos(out_features):
    return {
        'in': Identity(),
        'diag': (Diag_sum(), ['in']),
        'pe': (PositionalEncoding(out_features), ['diag'])
    }

def block(in_features, depth_of_mlp, constant_n_vertices=True):
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, in_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices), ['in']),
        'mlp2': (MlpBlock_Real(in_features, in_features, depth_of_mlp,
                constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(2*in_features, in_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices)
    }

def block_res(in_features, depth_of_mlp, constant_n_vertices=True):
    """Full residual block with two independent MLP branches and self-attention.

    Architecture:
        in → mlp1 ──┐
        in → mlp2 ──┤→ mult (matmul) → cat([mult, in]) → mlp3 → add([in, mlp3])

    Two separate MLP branches (mlp1, mlp2) are computed from the input and
    their matrix product is concatenated with the skip connection before a
    final projection.  A residual connection is added at the output.

    Parameter count: 3 × MLP(in, in, depth)
        - mlp1: MLP(in → in)
        - mlp2: MLP(in → in)
        - mlp3: MLP(2*in → in)

    Use when: accuracy is the priority and memory / compute budgets are ample.
    Compared to block_res_mem this variant uses one additional MLP branch,
    which can improve expressivity at the cost of ~50% more parameters.
    """
    return {
        'in': Identity(),
        'mlp1': (MlpBlock_Real(in_features, in_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices), ['in']),
        'mlp2': (MlpBlock_Real(in_features, in_features, depth_of_mlp,
                constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['mlp1', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(2*in_features, in_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices),
        'add': (Add(), ['in', 'mlp3'])
    }


def _block_res_mem_impl(in_features, depth_of_mlp, constant_n_vertices=True):
    """Internal implementation of the memory-efficient residual block."""
    return {
        'in': Identity(),
        'mlp2': (MlpBlock_Real(in_features, in_features, depth_of_mlp,
                constant_n_vertices=constant_n_vertices), ['in']),
        'mult': (Matmul(), ['in', 'mlp2']),
        'cat':  (Concat(), ['mult', 'in']),
        'mlp3': MlpBlock_Real(2*in_features, in_features, depth_of_mlp,
            constant_n_vertices=constant_n_vertices),
        'add': (Add(), ['in', 'mlp3'])
    }


def block_res_memory_efficient(in_features, depth_of_mlp, constant_n_vertices=True):
    """Memory-efficient residual block with self-gating.

    Architecture:
        in → mlp2 → h2 → in ⊗ h2 (matmul) → cat([mult, in]) → mlp3 → add([in, mlp3])

    A single MLP branch (mlp2) is computed from the input and its matrix
    product with the raw input is used as the gating signal.  The result is
    concatenated with the skip connection before a final projection, and a
    residual connection is added at the output.

    Parameter count: 2 × MLP(in, in, depth)  (~50% fewer than block_res)
        - mlp2: MLP(in → in)
        - mlp3: MLP(2*in → in)

    Use when: memory is constrained or when training large batches.
    """
    return _block_res_mem_impl(in_features, depth_of_mlp, constant_n_vertices)


# Backward-compatible alias
block_res_mem = block_res_memory_efficient

def base_model(num_blocks, in_features, depth_of_mlp, block, constant_n_vertices=True):
    d = {'in': Identity()}
    for i in range(num_blocks-1):
        d['block'+str(i+1)] = block(in_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    d['block'+str(num_blocks)] = block(in_features, depth_of_mlp, constant_n_vertices=constant_n_vertices)
    return d

def node_embedding_node_pos(original_features_num, num_blocks, 
                        in_features, depth_of_mlp,
                        block_inside, constant_n_vertices=True, **kwargs):
    d = {'in': Identity()}
    d['emb'] = init_emb(original_features_num, in_features)
    d['bm'] = base_model(num_blocks, in_features, depth_of_mlp, block_inside, constant_n_vertices=constant_n_vertices)
    d['bm_out'] = ColumnMaxPooling()
    d['skip'] = (Identitynn(), ['in'])
    d['node_emb'] = node_pos(in_features)
    d['node_out'] = Identitynn()
    d['suffix'] = (Concat(), ['bm_out', 'node_out'])
    return d