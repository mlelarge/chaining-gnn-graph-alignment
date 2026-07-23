#!/usr/bin/env python3
"""Render the reproduction JSONL into the README's markdown tables.

Each cell's record stores per-pair arrays under ``methods.<name>.{acc,nce}``
(older runs used flat ``<name>: [acc_mean, nce_mean]``). Means are computed from
the arrays; with ``--ci`` (and >1 sample) cells show ``mean±h`` where
``h = 1.96*std/sqrt(n)`` (95% CI half-width). ER-Reg accuracies are fractions;
real-world accuracies are percentages (matching the paper's conventions).

    python repro/format_tables.py [results.jsonl] [--ci]
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "results", "repro_seed0.jsonl")

ERREG_ROWS = [
    ("Proj(D_cx)", "proj_dcx"),
    ("FAQ(D_cx)", "faq_dcx"),
    ("BAPG-GW Proj", "bapg_proj"),
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


def _mean(a):
    return sum(a) / len(a)


def _pstd(a):
    m = _mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / len(a))


def _fmt_frac(a):
    return f"{a:.3f}" if a < 0.1 else f"{a:.2f}"


def _series(rec, key):
    """(acc_array, nce_array) for a method — new methods-schema or old flat means.

    Returns (None, None) when the method is absent from the record (e.g. a
    fresh reproduce_results run before the add_bapg merge)."""
    if "methods" in rec:
        m = rec["methods"].get(key)
        if m is None:
            return None, None
        return m.get("acc"), m["nce"]
    v = rec.get(key)
    if v is None:
        return None, None
    return [v[0]], [v[1]]


def cell(rec, key, ci, pct=False):
    acc, nce = _series(rec, key)
    if nce is None:
        return "–"
    a, m_nce = _mean(acc), _mean(nce)
    av = f"{a * 100:.1f}" if pct else _fmt_frac(a)
    if ci and len(acc) > 1:
        k = len(acc)
        ha = 1.96 * _pstd(acc) / math.sqrt(k) * (100 if pct else 1)
        hn = 1.96 * _pstd(nce) / math.sqrt(k)
        av = f"{av}±{ha:.1f}" if pct else f"{av}±{ha:.2f}"
        return f"{av}/{m_nce:.0f}±{hn:.0f}"
    return f"{av}/{m_nce:.0f}"


def _max_nce(rec):
    if "methods" in rec:
        return _mean(rec["methods"]["max"]["nce"])
    return rec["max_nce"]


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


def synthetic_table(family, cells, ci):
    noises = sorted(cells)
    head = "| p | " + " | ".join("%g" % p for p in noises) + " |"
    sep = "|---|" + "|".join("---" for _ in noises) + "|"
    lines = [FAMILY_TITLE[family], "", head, sep]
    for label, key in ERREG_ROWS:
        lines.append("| " + label + " | " + " | ".join(cell(cells[p], key, ci) for p in noises) + " |")
    return "\n".join(lines)


def real_table(real, ci):
    cols = [c for c in REAL_COLS if c[1] in real]
    head = "| Method | " + " | ".join(h for h, _ in cols) + " |"
    sep = "|---|" + "|".join("---" for _ in cols) + "|"
    lines = ["**Noisy real-world networks** (edge add/remove noise; tab:realworld-noisy)", "", head, sep]
    for label, key in [("FAQ(D_cx)", "faq_dcx"), ("ChFGNN-ER4", "chfgnn_er4"), ("ChFGNN", "chfgnn")]:
        lines.append("| " + label + " | " + " | ".join(cell(real[c], key, ci, pct=True) for _, c in cols) + " |")
    lines.append("| Max nce | " + " | ".join(f"– /{_max_nce(real[c]):.0f}" for _, c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Render reproduction JSONL to README tables.")
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--ci", action="store_true", help="Append 95%% CI half-widths (needs per-sample arrays).")
    args = ap.parse_args()
    erreg, real = load(args.path)
    for fam in ["sparse", "dense", "regular"]:
        if fam in erreg:
            print(synthetic_table(fam, erreg[fam], args.ci))
            print()
    if real:
        print(real_table(real, args.ci))


if __name__ == "__main__":
    main()
