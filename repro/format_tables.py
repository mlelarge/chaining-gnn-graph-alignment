#!/usr/bin/env python3
"""Render the reproduction JSONL (from reproduce_results.py) into the README's
markdown tables. ER-Reg accuracies are fractions; real-world accuracies are
percentages (matching the paper's table conventions).

    python repro/format_tables.py [results.jsonl]   # default repro/results/repro_seed0.jsonl
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "results", "repro_seed0.jsonl")

ERREG_ROWS = [
    ("Proj(D_cx)", "proj_dcx"),
    ("FAQ(D_cx)", "faq_dcx"),
    ("FGNN Proj", "fgnn_proj"),
    ("FGNN FAQ", "fgnn_faq"),
    ("ChFGNN Proj", "chfgnn_proj"),
    ("ChFGNN FAQ", "chfgnn_faq"),
]
FAMILY_TITLE = {
    "sparse": "**Sparse Erdős–Rényi, average degree 4** (nce_max ≈ 1000)",
    "dense": "**Dense Erdős–Rényi, average degree 80** (nce_max ≈ 20,000)",
    "regular": "**Regular graphs, degree 10** (nce_max = 2500)",
}
REAL_COLS = [  # header -> cell key (paper column order)
    ("yeast25LC 5%", "yeast25LC@0.05"),
    ("yeast25LC 10%", "yeast25LC@0.1"),
    ("ca-netscience 10%", "ca-netscience@0.1"),
    ("ca-netscience 20%", "ca-netscience@0.2"),
    ("inf-euroroad 10%", "inf-euroroad@0.1"),
    ("inf-euroroad 20%", "inf-euroroad@0.2"),
]


def fmt_frac(a):
    return f"{a:.3f}" if a < 0.1 else f"{a:.2f}"


def cell_frac(pair):
    a, n = pair
    return f"{fmt_frac(a)}/{n:.0f}"


def cell_pct(pair):
    a, n = pair
    return f"{a * 100:.1f}/{n:.0f}"


def load(path):
    erreg, real = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["table"] == "ER-Reg":
                erreg.setdefault(r["family"], {})[r["noise"]] = r
            else:
                real[r["cell"]] = r
    return erreg, real


def synthetic_table(family, cells):
    noises = sorted(cells)
    head = "| p | " + " | ".join("%g" % p for p in noises) + " |"
    sep = "|---|" + "|".join("---" for _ in noises) + "|"
    lines = [FAMILY_TITLE[family], "", head, sep]
    for label, key in ERREG_ROWS:
        lines.append("| " + label + " | " + " | ".join(cell_frac(cells[p][key]) for p in noises) + " |")
    return "\n".join(lines)


def real_table(real):
    cols = [c for c in REAL_COLS if c[1] in real]
    head = "| Method | " + " | ".join(h for h, _ in cols) + " |"
    sep = "|---|" + "|".join("---" for _ in cols) + "|"
    lines = ["**Noisy real-world networks** (edge add/remove noise; tab:realworld-noisy)", "", head, sep]
    for label, key in [("FAQ(D_cx)", "faq_dcx"), ("ChFGNN-ER4", "chfgnn_er4"), ("ChFGNN", "chfgnn")]:
        lines.append("| " + label + " | " + " | ".join(cell_pct(real[c][key]) for _, c in cols) + " |")
    lines.append("| Max nce | " + " | ".join(f"– /{real[c]['max_nce']:.0f}" for _, c in cols) + " |")
    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    erreg, real = load(path)
    for fam in ["sparse", "dense", "regular"]:
        if fam in erreg:
            print(synthetic_table(fam, erreg[fam]))
            print()
    if real:
        print(real_table(real))


if __name__ == "__main__":
    main()
