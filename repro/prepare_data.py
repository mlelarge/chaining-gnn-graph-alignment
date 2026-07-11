"""Prepare real-world graph-alignment datasets for reproduction (Phase 3).

A self-contained port of the dataset-creation notebooks in the FUGAL repo
(``canets_dataset``, ``road_dataset``, ``MultiMagna_dataset``). Turns the raw
edge lists committed under ``data/raw/`` into the train/val/test parquet pairs
the Hydra dataset configs consume, using the SAME helpers as the rest of the
codebase (``loaders.real_noise``), so the output is byte-compatible with the
loader (``loaders.__init__._get_real``).

Three pair constructions, matching the notebooks / paper tables:

* ``er_self``  — graph_A = G, graph_B = ER-noised G (edge add/remove at G's own
  average degree). Used for **ca-netscience** and **inf-euroroad**
  (tab:realworld-noisy). Noise names follow the notebooks: ``noise1`` = 0.1,
  ``noise2`` = 0.2.
* ``er_pair``  — graph_A = yeast0 (trusted base, 8,323 edges), graph_B =
  ER-noised yeast25 (the q=25% low-confidence variant). This is the harder
  **yeast25LC** benchmark (tab:realworld-noisy); ``noise005`` = 0.05,
  ``noise01`` = 0.1. yeast0/yeast25 share the same 1,004-node set.
* ``addition`` — graph_A = yeast0, graph_B = yeast_q (clean low-confidence
  variant, no extra noise). The **MultiMAGNA** edge-addition benchmark
  (tab:multimagna-full); the true correspondence is the identity and the max
  number of common edges is 8,323.

Reproducibility (resolved with the author): the paper used **no fixed seed** and
a **single noise realization** per cell, so ``--seed`` defaults to ``None``.

Output path mirrors the loader: ``<output-dir>/<data_subdir>/<name>.parquet``,
with ``--output-dir`` defaulting to ``<root-dir|~>/experiments-gnn-gap/data``
(the ``DATA_PB_DIR`` that ``commander.py`` derives), so the configs find them.

Usage::

    python -m repro.prepare_data --all
    python -m repro.prepare_data --dataset ca-netscience
    python -m repro.prepare_data --dataset multimagna-noisy --seed 0
"""

import argparse
import os

import numpy as np
import torch

from loaders.real_noise import get_adj, make_noisy, pairs_to_parquet
from loaders.representations import adjacency_matrix_to_tensor_representation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")


def _raw(rel):
    return os.path.join(RAW_DIR, rel)


def _repr(A):
    return adjacency_matrix_to_tensor_representation(torch.as_tensor(A, dtype=torch.float))


def load_shared(paths):
    """Read several edge lists onto one shared, contiguous node index (union of nodes)."""
    import pandas as pd

    edge_sets = []
    nodes = set()
    for p in paths:
        df = pd.read_csv(p, sep=r"\s+", header=None, usecols=[0, 1], names=["s", "t"])
        edges = list(zip(df["s"].tolist(), df["t"].tolist()))
        edge_sets.append(edges)
        nodes |= {n for e in edges for n in e}
    idx = {node: i for i, node in enumerate(sorted(nodes))}
    adjs = []
    for edges in edge_sets:
        A = np.zeros((len(idx), len(idx)), dtype=int)
        for s, t in edges:
            A[idx[s], idx[t]] = 1
            A[idx[t], idx[s]] = 1
        adjs.append(A)
    return adjs


# Output registry — one entry per parquet stem, named to match the dataset configs.
# noise tags follow the FUGAL notebooks (noise1=0.1, noise2=0.2, noise005=0.05, noise01=0.1).
DATASETS = {
    "ca-netscience": {
        "subdir": "ca-netscience",
        "n_train": 200, "n_val": 20,
        "outputs": [
            {"name": "ca_nets_noise1", "model": "er_self", "graph": "ca-netscience/ca-netscience.txt", "noise": 0.1},
            {"name": "ca_nets_noise2", "model": "er_self", "graph": "ca-netscience/ca-netscience.txt", "noise": 0.2},
        ],
    },
    "inf-euroroad": {
        "subdir": "inf-euroroad",
        "n_train": 20, "n_val": 5,
        "outputs": [
            {"name": "road_noise1", "model": "er_self", "graph": "inf-euroroad/inf-euroroad.txt", "noise": 0.1},
            {"name": "road_noise2", "model": "er_self", "graph": "inf-euroroad/inf-euroroad.txt", "noise": 0.2},
        ],
    },
    "multimagna-noisy": {  # harder yeast25LC: A=yeast0, B=noisy(yeast25)
        "subdir": "MultiMagna",
        "n_train": 20, "n_val": 20,
        "outputs": [
            {"name": "yeast0_25_noise005", "model": "er_pair", "graph_a": "MultiMagna/yeast0_Y2H1.txt", "graph_b": "MultiMagna/yeast25_Y2H1.txt", "noise": 0.05},
            {"name": "yeast0_25_noise01", "model": "er_pair", "graph_a": "MultiMagna/yeast0_Y2H1.txt", "graph_b": "MultiMagna/yeast25_Y2H1.txt", "noise": 0.1},
        ],
    },
    "multimagna-full": {  # tab:multimagna-full: A=yeast0, B=yeast_q (clean)
        "subdir": "MultiMagna",
        "n_train": 20, "n_val": 20,
        "outputs": [
            {"name": f"multimagna_yeast{q}", "model": "addition", "graph_a": "MultiMagna/yeast0_Y2H1.txt", "graph_b": f"MultiMagna/yeast{q}_Y2H1.txt"}
            for q in (5, 10, 15, 20, 25)
        ],
    },
}


def build_pairs(out, n, rng):
    """Build n (graph_A, graph_B) tensor-representation pairs for one output spec."""
    model = out["model"]
    if model == "er_self":
        A = get_adj(_raw(out["graph"]))
        info = f"{len(A)} nodes, avg deg {A.sum() / len(A):.2f}, ER noise {out['noise']}"
        return [(_repr(A), _repr(make_noisy(A, out["noise"], rng=rng))) for _ in range(n)], info
    if model == "er_pair":
        A, B = load_shared([_raw(out["graph_a"]), _raw(out["graph_b"])])
        info = f"n={len(A)} A_edges={A.sum() // 2} B_edges={B.sum() // 2}, ER noise {out['noise']} on B"
        return [(_repr(A), _repr(make_noisy(B, out["noise"], rng=rng))) for _ in range(n)], info
    if model == "addition":
        A, B = load_shared([_raw(out["graph_a"]), _raw(out["graph_b"])])
        info = f"n={len(A)} base_edges={A.sum() // 2} variant_edges={B.sum() // 2} (clean, identity perm)"
        return [(_repr(A), _repr(B)) for _ in range(n)], info
    raise ValueError(f"unknown model {model!r}")


def prepare(key, spec, output_dir, rng):
    out_subdir = os.path.join(output_dir, spec["subdir"])
    for out in spec["outputs"]:
        train, info = build_pairs(out, spec["n_train"], rng)
        test, _ = build_pairs(out, spec["n_val"], rng)
        print(f"[{key}] {out['name']}: {info}")
        pairs_to_parquet(train, os.path.join(out_subdir, f"{out['name']}_train.parquet"), rng=rng)
        pairs_to_parquet(test, os.path.join(out_subdir, f"{out['name']}_test.parquet"), rng=rng)
        print(
            f"    -> config: data_subdir={spec['subdir']}  "
            f"train.name={out['name']}_train  val/test.name={out['name']}_test"
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare real-world datasets (Phase 3).")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Which dataset to prepare.")
    parser.add_argument("--all", action="store_true", help="Prepare every dataset.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed. Default None reproduces the paper (no fixed seed).",
    )
    parser.add_argument(
        "--root-dir", default=None,
        help="Base for the default output dir (<root>/experiments-gnn-gap/data). Default: ~",
    )
    parser.add_argument("--output-dir", default=None, help="Override the output directory directly.")
    parser.add_argument("--n-train", type=int, default=None, help="Override the train-set size (default: paper size).")
    parser.add_argument("--n-val", type=int, default=None, help="Override the val/test-set size (default: paper size).")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("pass --dataset <name> or --all")

    rng = np.random.default_rng(args.seed)  # seed None -> non-deterministic (paper behavior)
    root = args.root_dir if args.root_dir is not None else os.path.expanduser("~")
    output_dir = args.output_dir or os.path.join(root, "experiments-gnn-gap", "data")
    print(f"Output dir: {output_dir}\n")

    for key in (sorted(DATASETS) if args.all else [args.dataset]):
        spec = dict(DATASETS[key])
        if args.n_train is not None:
            spec["n_train"] = args.n_train
        if args.n_val is not None:
            spec["n_val"] = args.n_val
        prepare(key, spec, output_dir, rng)


if __name__ == "__main__":
    main()
