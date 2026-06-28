#!/usr/bin/env python3
"""Per-sample analysis figure for the reproduction (needs the `viz` extra:
``uv sync --extra viz``).

Reads a per-sample reproduction JSONL (reproduce_results.py) and writes
``repro/results/per_sample_analysis.png`` with two panels:

  (a) Sparse ER — per-pair accuracy of each decoder vs noise. The means hide a
      *bimodal* transition (pairs are either solved or not); this shows it.
  (b) Paired scatter of ChFGNN-FAQ vs FAQ(D_cx) per-pair accuracy across all
      synthetic families, with the y=x line. No point falls below the line:
      ChFGNN-FAQ never loses a pair the convex baseline wins (strict dominance),
      and the off-diagonal top-left cloud is the pairs it rescues.

    python repro/plot_samples.py [results.jsonl] [-o out.png]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "results", "repro_seed0.jsonl")

M_COLOR = {"faq_dcx": "#d62728", "fgnn_faq": "#1f77b4", "chfgnn_faq": "#2ca02c"}
M_LABEL = {"faq_dcx": "FAQ(D_cx)", "fgnn_faq": "FGNN-FAQ", "chfgnn_faq": "ChFGNN-FAQ"}
FAM_COLOR = {"sparse": "#2ca02c", "dense": "#9467bd", "regular": "#ff7f0e"}


def load(path):
    cells = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["table"] == "ER-Reg":
                cells[(r["family"], r["noise"])] = r["methods"]
    return cells


def main():
    ap = argparse.ArgumentParser(description="Per-sample analysis figure.")
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "results", "per_sample_analysis.png"))
    args = ap.parse_args()
    cells = load(args.path)
    rng = np.random.default_rng(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3))

    # (a) sparse: per-pair accuracy of each decoder vs noise (jittered)
    noises = sorted(n for (fam, n) in cells if fam == "sparse")
    for mi, m in enumerate(["faq_dcx", "fgnn_faq", "chfgnn_faq"]):
        xs, ys = [], []
        for n in noises:
            acc = cells[("sparse", n)][m]["acc"]
            xs += [n + (mi - 1) * 0.011 + j for j in rng.normal(0, 0.004, len(acc))]
            ys += list(acc)
        ax1.scatter(xs, ys, s=15, alpha=0.5, color=M_COLOR[m], edgecolors="none", label=M_LABEL[m])
    ax1.set(xlabel="noise $p$", ylabel="per-pair accuracy", ylim=(-0.05, 1.05),
            title="(a) Sparse ER — each decoder's accuracy per pair")
    ax1.legend(loc="center left", fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.25)

    # (b) paired scatter ChFGNN-FAQ vs FAQ(D_cx), all synthetic families
    for (fam, n), m in cells.items():
        x = np.array(m["faq_dcx"]["acc"]) + rng.normal(0, 0.008, len(m["faq_dcx"]["acc"]))
        y = np.array(m["chfgnn_faq"]["acc"]) + rng.normal(0, 0.008, len(m["chfgnn_faq"]["acc"]))
        ax2.scatter(x, y, s=15, alpha=0.45, color=FAM_COLOR[fam], edgecolors="none")
    ax2.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1.2, label="$y = x$")
    for fam, c in FAM_COLOR.items():
        ax2.scatter([], [], color=c, label=fam)
    ax2.set(xlabel="FAQ(D_cx) per-pair accuracy", ylabel="ChFGNN-FAQ per-pair accuracy",
            xlim=(-0.05, 1.05), ylim=(-0.05, 1.05),
            title="(b) Paired — lower-right is empty:\nno pair FAQ(D_cx) solves that ChFGNN-FAQ misses")
    ax2.set_aspect("equal")
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
