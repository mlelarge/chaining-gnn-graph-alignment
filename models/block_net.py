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

def block_res_mem(in_features, depth_of_mlp, constant_n_vertices=True):
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