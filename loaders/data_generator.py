"""Backward compatibility shim. Import from specific modules instead."""

# Re-export generators
from loaders.generators import (
    GENERATOR_FUNCTIONS,
    generates,
    generate_erdos_renyi_netx,
    generate_bernoulli_uniform,
    generate_regular_graph_netx,
    NOISE_FUNCTIONS,
    noise,
    noise_erdos_renyi,
    noise_bernoulli,
    noise_edge_swap,
    is_swappable,
    do_swap,
)

# Re-export representations
from loaders.representations import (
    adjacency_matrix_to_tensor_representation,
    adjacency_matrix_to_tensor_representation_ind,
)

# Re-export dataset classes
from loaders.datasets import Base_Generator, GAP_Generator, all_perm

# Re-export chaining utilities
from loaders.chaining_utils import all_ind, make_data_from_ind, make_data_from_ind_label
