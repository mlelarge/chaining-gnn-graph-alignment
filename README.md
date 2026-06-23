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

Every results table maps to one `make` target — `make synthetic`, `make realworld`, or a single cell
like `make ca-netscience` (run `make help` for the list). The commands each target runs are below.

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

## Results — synthetic graphs

Accuracy / number of common edges (`acc / nce`) as a function of the noise `p`, from the paper's
Table (tab:ER-Reg). `Proj` and `FAQ` are post-processing decoders; `FGNN` is a single network and
`ChFGNN` the chained variant. Reproduce with `make synthetic` (or `make synthetic-sparse|dense|regular`).

**Sparse Erdős–Rényi, average degree 4** (nce_max ≈ 1000):

| p | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 |
|---|---|---|---|---|---|---|---|---|
| Proj(D_cx) | 0.98/997 | 0.97/950 | 0.90/853 | 0.59/499 | 0.23/195 | 0.09/130 | 0.04/115 | 0.02/112 |
| FAQ(D_cx) | 0.98/997 | 0.98/950 | 0.96/898 | 0.95/847 | 0.73/723 | 0.13/504 | 0.04/487 | 0.02/485 |
| FGNN Proj | 0.98/997 | 0.94/925 | 0.74/674 | 0.44/365 | 0.23/193 | 0.12/134 | 0.06/114 | 0.03/103 |
| FGNN FAQ | 0.98/997 | 0.98/950 | 0.96/898 | 0.95/847 | 0.81/755 | 0.24/535 | 0.07/494 | 0.03/485 |
| ChFGNN Proj | 0.98/997 | 0.98/950 | 0.96/898 | 0.94/845 | 0.91/790 | 0.82/720 | 0.49/549 | 0.08/367 |
| ChFGNN FAQ | 0.98/997 | 0.98/950 | 0.96/899 | 0.95/849 | 0.93/800 | 0.85/742 | 0.52/638 | 0.09/546 |

**Dense Erdős–Rényi, average degree 80** (nce_max ≈ 20,000):

| p | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 |
|---|---|---|---|---|---|---|---|---|
| Proj(D_cx) | 1.00/19964 | 1.00/18987 | 1.00/17966 | 0.61/8700 | 0.14/3888 | 0.04/3646 | 0.02/3633 | 0.01/3624 |
| FAQ(D_cx) | 1.00/19964 | 1.00/18987 | 1.00/17968 | 1.00/16990 | 1.00/15972 | 0.21/7922 | 0.01/6272 | 0.01/6276 |
| FGNN Proj | 1.00/19964 | 1.00/18979 | 0.73/11254 | 0.28/4674 | 0.10/3651 | 0.04/3521 | 0.02/3517 | 0.01/3505 |
| FGNN FAQ | 1.00/19964 | 1.00/18987 | 1.00/17968 | 1.00/16990 | 0.95/15390 | 0.14/7031 | 0.01/6259 | 0.01/6254 |
| ChFGNN Proj | 1.00/19964 | 1.00/18987 | 0.94/16241 | 0.83/13028 | 0.68/10291 | 0.37/6574 | 0.02/3690 | 0.01/3591 |
| ChFGNN FAQ | 1.00/19964 | 1.00/18987 | 1.00/17968 | 1.00/16990 | 0.99/15972 | 0.62/11577 | 0.01/6263 | 0.01/6255 |

**Regular graphs, degree 10** (nce_max = 2500):

| p | 0 | 0.05 | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|---|
| Proj(D_cx) | 0.002/51 | 0.002/51 | 0.003/50 | 0.001/49 | 0.002/50 |
| FAQ(D_cx) | 0.002/385 | 0.003/425 | 0.003/456 | 0.002/369 | 0.003/496 |
| FGNN Proj | 1.00/2500 | 0.31/405 | 0.03/113 | 0.005/108 | 0.003/106 |
| FGNN FAQ | 1.00/2500 | 0.95/2059 | 0.10/912 | 0.005/837 | 0.002/838 |
| ChFGNN Proj | 1.00/2500 | 0.95/2034 | 0.54/1135 | 0.009/281 | 0.003/95 |
| ChFGNN FAQ | 1.00/2500 | 0.95/2059 | 0.56/1383 | 0.008/871 | 0.003/836 |

## Results — real-world graphs

`acc / nce`. **ChFGNN-ER4** is the synthetic sparse-ER model transferred zero-shot
(`v1.0.0-er500-d4-pn0.22`); **ChFGNN** is the dataset-specific model (the `v1.1.0-*` releases). FUGAL
([idea-iitd/Fugal](https://github.com/idea-iitd/Fugal), `mu=1`) and SGWL are external baselines.
Reproduce with `make realworld`.

**Noisy real-world networks** (edge add/remove noise; tab:realworld-noisy):

| Method | yeast25LC 5% | yeast25LC 10% | ca-netscience 10% | ca-netscience 20% | inf-euroroad 10% | inf-euroroad 20% |
|---|---|---|---|---|---|---|
| FUGAL | 53.1/7480 | 44.6/7035 | 60.3/794 | 37.7/629 | 18.3/818 | 2.9/714 |
| FAQ(D_cx) | 49.8/7660 | 44.7/7245 | 65.2/822 | 45.6/687 | 55.8/1170 | 10.9/940 |
| ChFGNN-ER4 | 47.6/7693 | 42.3/7297 | 63.5/818 | 44.1/688 | 40.0/1111 | 7.5/970 |
| ChFGNN | 54.1/7732 | 51.3/7404 | 65.4/824 | 57.0/724 | 63.5/1213 | 15.4/963 |
| Max nce | – /7909 | – /7498 | – /826 | – /730 | – /1272 | – /1137 |

**MultiMAGNA yeast PPI** (edge-addition low-confidence variants; tab:multimagna-full). "training" =
the variant used to train the dataset-specific ChFGNN (not a test cell):

| Method | 5% conf | 10% conf | 15% conf | 20% conf | 25% conf |
|---|---|---|---|---|---|
| FAQ(J) | 37.5/7383 | 34.4/7245 | 29.1/6807 | 23.9/6689 | 36.4/7383 |
| SGWL | 83.6/– | – | 66.6/– | – | 58.8/– |
| FUGAL | 83.0/8311 | 77.7/8231 | 74.3/8172 | 70.9/8148 | 68.6/8095 |
| FAQ(D_cx) | 84.2/8323 | 82.6/8317 | 78.0/8289 | 77.0/8294 | 76.1/8306 |
| ChFGNN-ER4 | 80.3/8300 | 75.3/8288 | 67.2/8252 | 63.1/8213 | 53.1/8080 |
| ChFGNN | training | training | training | 72.2/8300 | 69.8/8291 |

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
├── Makefile                # One target per paper table (make help)
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
