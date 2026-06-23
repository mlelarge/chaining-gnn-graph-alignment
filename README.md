# Chaining 2-FWL GNNs for Combinatorial Graph Alignment

This repository contains the code for the paper [Chaining 2-FWL GNNs for Combinatorial Graph Alignment](https://arxiv.org/abs/2510.03086).

## The combinatorial Graph Alignment Problem (GAP)

![](assets/gap.gif)

Given two n×n adjacency matrices A and B, representing graphs G_A and G_B, the **graph alignment problem** aims to find the permutation π that best matches their structures by aligning corresponding edges. Formally, the objective is  
![alignment objective](https://latex.codecogs.com/svg.latex?\large\max_{\pi\in\mathcal{S}_n}\sum_{i,j}A_{ij}B_{\pi(i)\pi(j)})


To evaluate the quality of an alignment, we define the **number of common edges** under a permutation π as  
![nce formula](https://latex.codecogs.com/svg.latex?\large\mathbf{nce}(\pi)=\frac12\sum_{i,j}A_{ij}B_{\pi(i)\pi(j)})

The factor of 1/2 corrects for double-counting edges in undirected graphs.

## Chained FGNNs

Starting from input graphs G_A and G_B, we first (1) extract features and compute similarities, then iteratively (2) rank nodes by alignment quality, and (3) use rankings to enhance features and similarities.

![](assets/chaining.png)

### The D_cx initialization (FAQ)

A recurring theme of the paper is that the classical **FAQ** solver is much stronger than recently reported once it is *initialized well*. We distinguish two initializations:

- **FAQ(J)** — FAQ started from the uninformative barycenter `J = 1·1ᵀ/n` (scipy's default).
- **FAQ(D_cx)** — FAQ started from the solution of the **convex** relaxation
  `min ‖A P − P B‖²_F` over doubly-stochastic matrices, solved by Frank–Wolfe.

The convex (`D_cx`) solver is isolated in [`toolbox/frank_wolfe.py`](toolbox/frank_wolfe.py)
(`relaxed_normAPPB_FW_seeds` / `solve_dcx`). The `FAQ(D_cx)`-vs-`FAQ(J)` gap is reproduced by
[`run_baseline.py`](run_baseline.py); chained FGNNs are then trained to improve on the strengthened
`FAQ(D_cx)` baseline.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for a reproducible environment (Python ≥ 3.12):

```bash
git clone <repo-url>
cd chaining-gnn-graph-alignment
uv sync            # creates .venv from the pinned uv.lock
```

Run scripts through the environment, e.g. `uv run python run_inference.py --help`, or activate the
venv with `source .venv/bin/activate`. A plain `pip install -e .` into a Python ≥ 3.12 environment
also works as a fallback. All inference below runs on **CPU/MPS** — no GPU required.

## Quickstart — reproduce a result with no GPU

Pretrained models are distributed as GitHub Releases and downloaded on demand.

```bash
# Synthetic (sparse Erdős–Rényi, avg degree 4):
python run_inference.py      --release v1.0.0-er500-d4-pn0.22

# Real-world (ca-netscience): build the test data once, then reproduce:
python -m repro.prepare_data --dataset ca-netscience --output-dir ./data
python run_inference_real.py --release v1.1.0-canetscience-pn0.1 --data-dir ./data
```

## Reproducing the paper results

### Synthetic graphs

`run_inference.py` downloads a pretrained chained FGNN and runs the inference loop; `run_baseline.py`
computes the FAQ baselines on the same cached data.

```bash
python run_inference.py --release v1.0.0-er500-d4-pn0.22      # sparse ER (d=4)
python run_inference.py --release v1.0.0-er500-d80-pn0.24     # dense ER  (d=80)
python run_inference.py --release v1.0.0-reg500-d10-pn0.11    # regular   (d=10)

python run_baseline.py  --release v1.0.0-er500-d4-pn0.22      # FAQ(D_cx) vs FAQ(J) + Max-nce
```

### Real-world graphs

The three real-world benchmarks (yeast PPI, ca-netscience coauthorship, inf-euroroad road network)
ship as tiny raw edge lists under [`data/raw/`](data/raw/) (see [`data/raw/SOURCES.md`](data/raw/SOURCES.md)
for provenance and citations). `repro/prepare_data.py` turns them into the train/val/test parquet
pairs the pipeline consumes (two noise models, matching the paper); `run_inference_real.py` downloads
a pretrained model and reports `acc / nce`.

```bash
# 1. build the real-world datasets (all of them) from the committed raw edge lists:
python -m repro.prepare_data --all --output-dir ./data

# 2. reproduce a table cell (downloads the matching pretrained model):
python run_inference_real.py --release v1.1.0-canetscience-pn0.1 --data-dir ./data
python run_inference_real.py --release v1.1.0-euroroad-pn0.2     --data-dir ./data
python run_inference_real.py --release v1.1.0-yeast25lc-pn0.05   --data-dir ./data
python run_inference_real.py --release v1.1.0-multimagna         --data-dir ./data --test-name multimagna_yeast20_test
```

The published numbers were produced with **no fixed seed** and a **single noise realization** per
cell, so `prepare_data` defaults `--seed` to `None`; expect small run-to-run variation (the paper
notes `nce` is the more reliable metric, as the base graphs have large automorphism groups).

### Pretrained checkpoints

| Release tag | Model |
|---|---|
| `v1.0.0-er500-d4-pn0.22` | ChFGNN-ER4 (sparse ER, also the real-world *transfer* model) |
| `v1.0.0-er500-d80-pn0.24` | ChFGNN, dense ER |
| `v1.0.0-reg500-d10-pn0.11` | ChFGNN, regular graphs |
| `v1.1.0-canetscience-pn0.1` / `-pn0.2` | ChFGNN, ca-netscience (noise 0.1 / 0.2) |
| `v1.1.0-euroroad-pn0.1` / `-pn0.2` | ChFGNN, inf-euroroad (noise 0.1 / 0.2) |
| `v1.1.0-yeast25lc-pn0.05` / `-pn0.1` | ChFGNN, yeast25LC (noise 0.05 / 0.1) |
| `v1.1.0-multimagna` | ChFGNN, MultiMAGNA yeast |

### Baselines (FAQ, FUGAL, SGWL)

`run_baseline.py` reproduces the in-repo FAQ baselines — **FAQ(D_cx)**, **FAQ(J)**, and the
**Max-nce** ceiling (FAQ seeded from the true permutation). The external baselines are **not**
vendored: **FUGAL** ([idea-iitd/Fugal](https://github.com/idea-iitd/Fugal), `mu=1`) and **SGWL**
(Xu et al., 2019) numbers come from their authors' code. Note that the FAQ numbers reported by some
prior work use the barycenter `J` initialization; initializing FAQ from the convex relaxation
(`FAQ(D_cx)`) already surpasses those.

## Training from scratch

Training uses [Hydra](https://hydra.cc/) configs in [`conf/`](conf/). By default, data and
checkpoints live under `~/experiments-gnn-gap/` (override with `root_dir=/path`).

```bash
python commander.py dataset=sparse                 # synthetic sparse ER
python commander.py dataset=ca_netscience          # real-world (after prepare_data)
```

## Performances on Synthetic datasets

Number of common edges (higher is better) for sparse Erdős-Rényi random graphs: 
| noise        | 0   | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 |
|--------------|-----|------|-----|------|-----|------|-----|------|
| Proj(D_cx) | 997 | 950 | 853 | 499 | 195 | 130 | 115 | 112 |
| FAQ(D_cx)  | 997 | 950 | 898 | 847 | 723 | 504 | 487 | 485 |
| ChFGNN Proj | 997 | 950 | 898 | 845 | 790 | 694 | 503 | 319 |
| ChFGNN FAQ  | 997 | 950 | 899 | 849 | 800 | 730 | 626 | 534 |

Number of common edges (higher is better) for dense Erdős-Rényi random graphs: 

| noise        | 0     | 0.05  | 0.1   | 0.15  | 0.2   | 0.25  | 0.3   | 0.35  |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Proj(D_cx) | 19964 | 18987 | 17966 | 8700  | 3888  | 3646  | 3633  | 3624  |
| FAQ(D_cx)  | 19964 | 18987 | 17968 | 16990 | 15972 | 7922  | 6272  | 6276  |
| ChFGNN Proj | 19964 | 18969 | 16241 | 13028 | 9561  | 6166  | 3615  | 3591  |
| ChFGNN FAQ  | 19964 | 18987 | 17968 | 16990 | 15779 | 11227 | 6258  | 6255  |

Number of common edges (higher is better) for regular random graphs:

| noise        | 0    | 0.05 | 0.1  | 0.15 | 0.2  |
|--------------|------|------|------|------|------|
| Proj(D_cx) | 51   | 51   | 50   | 49   | 50   |
| FAQ(D_cx)  | 385  | 425  | 456  | 369  | 496  |
| ChFGNN Proj | 2500 | 1343 | 563  | 192  | 114  |
| ChFGNN FAQ  | 2500 | 2059 | 1438 | 850  | 837  |

## Project Structure

```
chaining-gnn-graph-alignment/
├── models/                 # Core model implementations
│   ├── pipeline.py         # Chaining pipeline (training + inference loop)
│   ├── pl_model.py         # Siamese network (PyTorch Lightning)
│   └── ...                 # FGNN / message-passing layers
├── loaders/                # Data generation and loading
│   ├── generators.py       # Synthetic graph generation + ER noise model
│   ├── real_noise.py       # Real-graph noising helpers
│   └── ...
├── toolbox/                # Metrics, baselines, the D_cx solver
│   ├── frank_wolfe.py      # Convex (D_cx) Frank–Wolfe solver
│   ├── baselines.py        # FAQ baselines (evaluate_faq_inits: D_cx vs J vs Max-nce)
│   └── metrics.py          # Evaluation metrics
├── repro/                  # Reproduction tooling
│   └── prepare_data.py     # raw edge lists -> train/val/test parquets
├── conf/                   # Hydra configuration files (config, dataset, model, training, pipeline)
├── data/raw/               # Committed raw real-world edge lists (+ SOURCES.md)
├── commander.py            # Training entry point (chaining)
├── run_inference.py        # Synthetic inference from a release
├── run_inference_real.py   # Real-world inference from a release
└── run_baseline.py         # FAQ baselines (D_cx vs J + Max-nce)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lelarge2025chaining,
  title={Chaining 2-FWL GNNs for Combinatorial Graph Alignment},
  author={Lelarge, Marc},
  journal={arXiv preprint arXiv:2510.03086},
  year={2025},
  url={https://arxiv.org/abs/2510.03086}
}
```

## Acknowledgments

We are grateful to the CLEPS infrastructure from the Inria of Paris for providing resources and support.
This project was provided with computing HPC and storage resources by GENCI at IDRIS thanks to the grant 2025-AD010613995R2 on the supercomputer Jean Zay's A100 and H100 partitions. 
