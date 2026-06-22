"""
Generate training and test parquet files for QAPlib instances.

Creates noisy versions of QAPlib (A, B) pairs as training data,
and the original pair as test data, in the format expected by
Base_Generator.

Usage:
    python scripts/make_qaplib_data.py --instances tai40a
    python scripts/make_qaplib_data.py --instances tai40a chr25a
    python scripts/make_qaplib_data.py --all --max-size 50
    python scripts/make_qaplib_data.py --instances tai40a --noise-levels 0.0 0.05 0.1
    python scripts/make_qaplib_data.py --instances tai40a --samples-per-noise 300
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch


def read_dat_file(filepath):
    """Read a QAPlib .dat file. Returns (n, A, B)."""
    with open(filepath, "r") as f:
        content = f.read()
    tokens = content.split()
    n = int(tokens[0])
    tokens = tokens[1:]
    dtype = int if "." not in tokens[0] else float
    values = [dtype(t) for t in tokens]
    A = np.array(values[: n * n]).reshape(n, n)
    B = np.array(values[n * n : 2 * n * n]).reshape(n, n)
    return n, A, B


def add_symmetric_noise(M, noise_prob, rng=None):
    """Add symmetric +-1 noise to off-diagonal entries with given probability."""
    if rng is None:
        rng = np.random.default_rng()
    n = M.shape[0]
    M_noisy = M.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < noise_prob:
                noise = rng.choice([-1, 1])
                M_noisy[i, j] += noise
                M_noisy[j, i] += noise
    return M_noisy


def matrix_to_tensor(M):
    """Convert a numpy matrix to the 2-channel tensor representation (2, n, n).

    Channel 0 is the matrix values, channel 1 is zeros (no positional encoding).
    """
    M_t = torch.as_tensor(M, dtype=torch.float)
    n = M_t.shape[0]
    B = torch.zeros((2, n, n))
    B[0, :, :] = M_t
    return B


def generate_instance_data(
    A,
    B,
    noise_levels,
    samples_per_noise,
    rng=None,
):
    """Generate noisy training pairs for one QAPlib instance.

    Returns a list of (tensor_A, tensor_B) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = []
    for noise in noise_levels:
        for _ in range(samples_per_noise):
            if noise == 0.0:
                A_noisy, B_noisy = A, B
            else:
                A_noisy = add_symmetric_noise(A, noise, rng)
                B_noisy = add_symmetric_noise(B, noise, rng)
            data.append((matrix_to_tensor(A_noisy), matrix_to_tensor(B_noisy)))
    return data


def save_parquet(data, path):
    """Save list of (tensor_A, tensor_B) as parquet."""
    structured = []
    for t_A, t_B in data:
        structured.append(
            {
                "graph_A": t_A.tolist(),
                "graph_B": t_B.tolist(),
            }
        )
    df = pd.DataFrame(structured)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(data)} samples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QAPlib training/test parquets"
    )
    parser.add_argument(
        "--data-dir",
        default="/lustre/fsn1/projects/rech/tdm/uuz44ie/experiments-gnn-gap/data_nl/",
        help="Path to QAPlib .dat files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for parquets (default: <data-dir>/parquet_data/)",
    )
    parser.add_argument(
        "--instances", nargs="+", default=None, help="Instance names (e.g. tai40a)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all .dat files in data-dir"
    )
    parser.add_argument(
        "--max-size", type=int, default=None, help="Skip instances with n > max-size"
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        help="Noise probabilities for training data",
    )
    parser.add_argument(
        "--samples-per-noise",
        type=int,
        default=200,
        help="Number of samples per noise level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, "parquet_data")
    rng = np.random.default_rng(args.seed)

    # Collect instances
    if args.all:
        dat_files = sorted(glob.glob(os.path.join(args.data_dir, "*.dat")))
        instances = [os.path.splitext(os.path.basename(f))[0] for f in dat_files]
    elif args.instances:
        instances = args.instances
    else:
        parser.error("Specify --instances or --all")

    for name in instances:
        filepath = os.path.join(args.data_dir, f"{name}.dat")
        if not os.path.exists(filepath):
            print(f"[SKIP] {name}: file not found at {filepath}")
            continue

        n, A, B = read_dat_file(filepath)
        if args.max_size and n > args.max_size:
            print(f"[SKIP] {name}: n={n} > {args.max_size}")
            continue

        print(f"[GEN]  {name} (n={n})")

        # Training data: noisy pairs
        train_data = generate_instance_data(
            A, B, args.noise_levels, args.samples_per_noise, rng
        )
        save_parquet(train_data, os.path.join(output_dir, f"{name}_noise.parquet"))

        # Test data: single clean pair
        test_data = [(matrix_to_tensor(A), matrix_to_tensor(B))]
        save_parquet(test_data, os.path.join(output_dir, f"{name}_test.parquet"))


if __name__ == "__main__":
    main()
