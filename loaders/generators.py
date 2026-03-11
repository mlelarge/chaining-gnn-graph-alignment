"""Graph generation and noise model registries."""

from typing import TypedDict, Optional
import networkx
import networkx as nx
import torch
import numpy as np
import random
import itertools


class GeneratorOutput(TypedDict):
    """Output of a single graph-pair generation call."""
    graph_A: nx.Graph
    adjacency_A: np.ndarray
    graph_B: Optional[nx.Graph]
    adjacency_B: np.ndarray
    permutation: np.ndarray
    edge_density: float
    noise_level: float


GENERATOR_FUNCTIONS = {}


def generates(name):
    """Register a generator function for a graph distribution"""

    def decorator(func):
        GENERATOR_FUNCTIONS[name] = func
        return func

    return decorator


@generates("ErdosRenyi")
def generate_erdos_renyi_netx(p, N, rng: Optional[np.random.Generator] = None):
    """Generate random Erdos Renyi graph"""
    g = networkx.erdos_renyi_graph(N, p)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float), p


@generates("Bernoulli")
def generate_bernoulli_uniform(a, N, rng: Optional[np.random.Generator] = None):
    # attention a is not the edge density!
    rng = rng or np.random.default_rng()
    edge_prob = rng.uniform(a, 1 - a, size=(N, N))
    edge_u = rng.random((N, N))
    return None, torch.as_tensor(edge_u < edge_prob, dtype=torch.float), edge_prob


@generates("Regular")
def generate_regular_graph_netx(p, N, rng: Optional[np.random.Generator] = None):
    """Generate random regular graph"""
    d = p * N
    d = int(d)
    # Make sure N * d is even
    if N * d % 2 == 1:
        d += 1
    g = networkx.random_regular_graph(d, N)
    W = networkx.adjacency_matrix(g).todense()
    return g, torch.as_tensor(W, dtype=torch.float), p


NOISE_FUNCTIONS = {}


def noise(name):
    """Register a noise function"""

    def decorator(func):
        NOISE_FUNCTIONS[name] = func
        return func

    return decorator


@noise("ErdosRenyi")
def noise_erdos_renyi(g, W, noise, edge_density, rng: Optional[np.random.Generator] = None):
    if edge_density >= 1.0:
        raise ValueError(f"edge_density must be < 1.0, got {edge_density}")
    n_vertices = len(W)
    pe1 = noise
    pe2 = min((edge_density * noise) / (1 - edge_density), 1.0)
    _, noise1, _ = generate_erdos_renyi_netx(pe1, n_vertices, rng=rng)
    _, noise2, _ = generate_erdos_renyi_netx(pe2, n_vertices, rng=rng)
    W_noise = W * (1 - noise1) + (1 - W) * noise2
    return W_noise


@noise("Bernoulli")
def noise_bernoulli(g, A, noise, edge_density, rng: Optional[np.random.Generator] = None):
    # Create an empty n x n adjacency matrix filled with zeros
    rng = rng or np.random.default_rng()
    r = 1 - noise
    edge_prob = (1 - r) * edge_density + r * A.numpy()
    N = A.shape[0]
    edge_u = rng.random((N, N))
    return torch.as_tensor(edge_u < edge_prob, dtype=torch.float)


def is_swappable(g, u, v, s, t):
    """
    Check whether we can swap
    the edges u,v and s,t
    to get u,t and s,v
    """
    actual_edges = g.has_edge(u, v) and g.has_edge(s, t)
    no_self_loop = (u != t) and (s != v)
    no_parallel_edge = not (g.has_edge(u, t) or g.has_edge(s, v))
    return actual_edges and no_self_loop and no_parallel_edge


def do_swap(g, u, v, s, t):
    g.remove_edge(u, v)
    g.remove_edge(s, t)
    g.add_edge(u, t)
    g.add_edge(s, v)


@noise("EdgeSwap")
def noise_edge_swap(g, W, noise, edge_density, rng: Optional[np.random.Generator] = None):  # Permet de garder la regularite
    g_noise = g.copy()
    edges_iter = list(itertools.chain(iter(g.edges), ((v, u) for (u, v) in g.edges)))
    for u, v in edges_iter:
        if random.random() < noise:
            for s, t in edges_iter:
                if random.random() < noise and is_swappable(g_noise, u, v, s, t):
                    do_swap(g_noise, u, v, s, t)
    W_noise = networkx.adjacency_matrix(g_noise).todense()
    return torch.as_tensor(W_noise, dtype=torch.float)
