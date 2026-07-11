"""Generate noisy real-graph datasets for the graph alignment pipeline.

Reads a real graph from an edge-list file, creates noisy copies with
Erdős-Rényi noise, applies random permutations, and saves train/test
splits as parquet files compatible with Base_Generator.

Usage:
    python -m loaders.real_noise \
        --edge-list /path/to/graph.txt \
        --noise 0.1 \
        --num-train 40 \
        --num-test 5 \
        --output-dir ~/experiments-gnn-gap/data/inf-euroroad/
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from more_itertools import chunked

from loaders.datasets import all_perm
from loaders.generators import noise_erdos_renyi
from loaders.representations import adjacency_matrix_to_tensor_representation


def get_adj(file):
    """Read an edge-list file and return a symmetric adjacency matrix."""
    df = pd.read_csv(file, sep=" ", header=None, names=["source", "target"])
    nodes = sorted(set(df["source"]) | set(df["target"]))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)

    for _, row in df.iterrows():
        i, j = node_to_idx[row["source"]], node_to_idx[row["target"]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix


def make_noisy(A, noise_level, rng=None):
    """Apply Erdős-Rényi noise to adjacency matrix A."""
    W = torch.as_tensor(A, dtype=torch.float)
    edge_density = W.sum().item() / (len(W) ** 2)
    return noise_erdos_renyi(None, W, noise_level, edge_density, rng=rng)


def generate_pairs(A, num_examples, noise_level, rng=None):
    """Generate num_examples noisy graph pairs from adjacency matrix A."""
    pairs = []
    for _ in range(num_examples):
        B_clean = adjacency_matrix_to_tensor_representation(
            torch.as_tensor(A, dtype=torch.float)
        )
        B_noisy = adjacency_matrix_to_tensor_representation(
            make_noisy(A, noise_level, rng=rng)
        )
        pairs.append((B_clean, B_noisy))
    return pairs


def pairs_to_parquet(pairs, path, rng=None):
    """Apply random permutations and save as parquet."""
    data = all_perm(chunked(iter(pairs), 1), rng=rng)
    structured_data = []
    for item in data:
        structured_data.append(
            {
                "graph_A": item[0].tolist(),
                "graph_B": item[1].tolist(),
                "permutation": item[2].tolist(),
            }
        )
    df = pd.DataFrame(structured_data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(data)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy real-graph datasets for graph alignment"
    )
    parser.add_argument(
        "--edge-list", required=True, help="Path to edge-list file"
    )
    parser.add_argument(
        "--noise", type=float, default=0.1, help="Noise level (default: 0.1)"
    )
    parser.add_argument(
        "--num-train", type=int, default=40, help="Number of training examples"
    )
    parser.add_argument(
        "--num-test", type=int, default=5, help="Number of test examples"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for output parquet files"
    )
    parser.add_argument(
        "--train-name", default=None, help="Train parquet filename stem (default: auto)"
    )
    parser.add_argument(
        "--test-name", default=None, help="Test parquet filename stem (default: auto)"
    )
    args = parser.parse_args()

    # Derive default names from the edge-list filename
    base = os.path.splitext(os.path.basename(args.edge_list))[0]
    noise_tag = str(args.noise).replace(".", "")
    train_name = args.train_name or f"{base}_noise{noise_tag}_train"
    test_name = args.test_name or f"{base}_noise{noise_tag}_test"

    print(f"Loading graph from {args.edge_list}")
    A = get_adj(args.edge_list)
    print(f"Graph has {len(A)} nodes, edge density {A.sum() / len(A)**2:.4f}")

    print(f"Generating {args.num_train} training examples (noise={args.noise})")
    train_pairs = generate_pairs(A, args.num_train, args.noise)
    pairs_to_parquet(
        train_pairs, os.path.join(args.output_dir, f"{train_name}.parquet")
    )

    print(f"Generating {args.num_test} test examples (noise={args.noise})")
    test_pairs = generate_pairs(A, args.num_test, args.noise)
    pairs_to_parquet(
        test_pairs, os.path.join(args.output_dir, f"{test_name}.parquet")
    )


if __name__ == "__main__":
    main()
