"""Prepare real-world graph-alignment datasets for reproduction (Phase 3).

Turns the committed raw edge lists under ``data/raw/`` into the train/val/test
parquet pairs the Hydra dataset configs consume. Two noise models, matching the
paper:

* **ER add/remove** (``model="er"``): corrupt the real graph with the
  edge-addition-removal noise model at the graph's own average degree
  (``loaders.generators.noise_erdos_renyi``). Used for ca-netscience,
  inf-euroroad, and the harder yeast benchmark (``yeast25LC`` in the paper).
* **MultiMAGNA edge-addition** (``model="addition"``): pair the trusted yeast
  base graph (``yeast0``) with a low-confidence variant
  (``yeast_q`` = base + q% added edges) on a shared node set. The true
  correspondence is the identity and the max number of common edges is the base
  edge count (8,323).

Reproducibility knobs (resolved with the author): the paper used **no fixed seed**
and a **single noise realization** per cell. ``--seed`` therefore defaults to
``None`` (paper behavior); pass any int for a deterministic rebuild.

Output path mirrors the loader (``loaders/__init__.py:_get_real``):
``<output-dir>/<data_subdir>/<name>.parquet`` where ``--output-dir`` defaults to
``<root-dir|~>/experiments-gnn-gap/data`` — the same ``DATA_PB_DIR`` that
``commander.py`` derives — so the dataset configs find the parquets.

Usage::

    python -m repro.prepare_data --all
    python -m repro.prepare_data --dataset ca-netscience --noise 0.1 0.2
    python -m repro.prepare_data --dataset multimagna --seed 0
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from loaders.real_noise import generate_pairs, get_adj, pairs_to_parquet
from loaders.representations import adjacency_matrix_to_tensor_representation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

# Dataset registry — paper specs. n_train/n_val from tab:realworld_stats;
# default noise levels from tab:realworld-noisy.
DATASETS = {
    "ca-netscience": {
        "subdir": "ca-netscience",
        "model": "er",
        "raw": "ca-netscience/ca-netscience.txt",
        "noises": [0.10, 0.20],
        "n_train": 200,
        "n_val": 20,
    },
    "inf-euroroad": {
        "subdir": "inf-euroroad",
        "model": "er",
        "raw": "inf-euroroad/inf-euroroad.txt",
        "noises": [0.10, 0.20],
        "n_train": 20,
        "n_val": 5,
    },
    # Harder "yeast25LC" benchmark: ER noise on the trusted base graph (8,323 edges).
    "yeast": {
        "subdir": "MultiMagna",
        "model": "er",
        "raw": "MultiMagna/yeast0_Y2H1.txt",
        "noises": [0.05, 0.10],
        "n_train": 20,
        "n_val": 20,
    },
    # MultiMAGNA edge-addition: (yeast0, yeast_q) pairs, identity correspondence.
    "multimagna": {
        "subdir": "MultiMagna",
        "model": "addition",
        "base": "MultiMagna/yeast0_Y2H1.txt",
        "variants": {q: f"MultiMagna/yeast{q}_Y2H1.txt" for q in (5, 10, 15, 20, 25)},
        "n_train": 20,
        "n_val": 20,
    },
}


def _read_edges(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["s", "t"])
    return list(zip(df["s"].tolist(), df["t"].tolist()))


def _adjacency_from_edges(edges, node_to_idx):
    n = len(node_to_idx)
    A = np.zeros((n, n), dtype=int)
    for s, t in edges:
        i, j = node_to_idx[s], node_to_idx[t]
        A[i, j] = 1
        A[j, i] = 1
    return A


def load_shared(paths):
    """Read several edge lists onto one shared, contiguous node index (union of nodes)."""
    edge_sets = [_read_edges(p) for p in paths]
    nodes = sorted({n for edges in edge_sets for e in edges for n in e})
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    return [_adjacency_from_edges(edges, node_to_idx) for edges in edge_sets]


def _pct(noise):
    return f"{int(round(noise * 100)):02d}"


def prepare_er(key, spec, noises, output_dir, rng):
    raw = os.path.join(RAW_DIR, spec["raw"])
    A = get_adj(raw)
    deg = A.sum() / len(A)
    print(f"[{key}] {len(A)} nodes, avg degree {deg:.2f}  (ER add/remove noise)")
    out_subdir = os.path.join(output_dir, spec["subdir"])
    for noise in noises:
        tag = f"{key}_q{_pct(noise)}"
        train = generate_pairs(A, spec["n_train"], noise, rng=rng)
        test = generate_pairs(A, spec["n_val"], noise, rng=rng)
        pairs_to_parquet(train, os.path.join(out_subdir, f"{tag}_train.parquet"), rng=rng)
        pairs_to_parquet(test, os.path.join(out_subdir, f"{tag}_test.parquet"), rng=rng)
        print(
            f"    -> config: data_subdir={spec['subdir']}  "
            f"train.name={tag}_train  val/test.name={tag}_test"
        )


def prepare_addition(key, spec, output_dir, rng):
    base_path = os.path.join(RAW_DIR, spec["base"])
    out_subdir = os.path.join(output_dir, spec["subdir"])
    for q, vpath in spec["variants"].items():
        A_base, A_var = load_shared([base_path, os.path.join(RAW_DIR, vpath)])
        print(
            f"[{key} q={q}] n={len(A_base)}  base_edges={A_base.sum() // 2}  "
            f"variant_edges={A_var.sum() // 2}  (edge-addition, identity true perm)"
        )
        repr_base = adjacency_matrix_to_tensor_representation(
            torch.as_tensor(A_base, dtype=torch.float)
        )
        repr_var = adjacency_matrix_to_tensor_representation(
            torch.as_tensor(A_var, dtype=torch.float)
        )
        tag = f"multimagna_yeast{q}"
        train = [(repr_base, repr_var) for _ in range(spec["n_train"])]
        test = [(repr_base, repr_var) for _ in range(spec["n_val"])]
        pairs_to_parquet(train, os.path.join(out_subdir, f"{tag}_train.parquet"), rng=rng)
        pairs_to_parquet(test, os.path.join(out_subdir, f"{tag}_test.parquet"), rng=rng)
        print(
            f"    -> config: data_subdir={spec['subdir']}  "
            f"train.name={tag}_train  val/test.name={tag}_test"
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare real-world datasets (Phase 3).")
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), help="Which dataset to prepare."
    )
    parser.add_argument("--all", action="store_true", help="Prepare every dataset.")
    parser.add_argument(
        "--noise", type=float, nargs="+", default=None,
        help="Override the ER noise level(s) (ignored for the multimagna addition model).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed. Default None reproduces the paper (no fixed seed).",
    )
    parser.add_argument(
        "--root-dir", default=None,
        help="Base for the default output dir (<root>/experiments-gnn-gap/data). Default: ~",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override the output directory directly."
    )
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("pass --dataset <name> or --all")

    rng = np.random.default_rng(args.seed)  # seed None -> non-deterministic (paper behavior)

    root = args.root_dir if args.root_dir is not None else os.path.expanduser("~")
    output_dir = args.output_dir or os.path.join(root, "experiments-gnn-gap", "data")
    print(f"Output dir: {output_dir}\n")

    keys = sorted(DATASETS) if args.all else [args.dataset]
    for key in keys:
        spec = DATASETS[key]
        if spec["model"] == "er":
            noises = args.noise if args.noise is not None else spec["noises"]
            prepare_er(key, spec, noises, output_dir, rng)
        else:
            prepare_addition(key, spec, output_dir, rng)


if __name__ == "__main__":
    main()
