"""Read inference .npy result files and print a Markdown table.

Metrics are rows, noise levels are columns.

Usage:
    python make_table.py results/new_results_L10.npy [results2.npy ...]

Single file: one table per file.
Multiple files: one table per metric, models as row groups.
"""

import sys
import numpy as np


def load_results(path):
    with open(path, "rb") as f:
        list_noises = np.load(f)
        ALL_acc     = np.load(f)   # (n_noises, n_ex) — FAQ acc
        ALL_qap     = np.load(f)   # (n_noises, n_ex) — FAQ common edges
        ALL_acc_p   = np.load(f)   # (n_noises, n_ex) — LAP acc
        ALL_qap_p   = np.load(f)   # (n_noises, n_ex) — LAP common edges
        ALL_acc_max = np.load(f)   # (n_noises, n_ex) — argmax acc
        ALL_nit     = np.load(f)   # (n_noises, n_ex) — FAQ iterations
        ALL_nloop   = np.load(f)   # (n_noises, n_ex) — chain loops used
    return dict(
        noises=list_noises,
        acc=ALL_acc,
        qap=ALL_qap,
        acc_p=ALL_acc_p,
        qap_p=ALL_qap_p,
        acc_max=ALL_acc_max,
        nit=ALL_nit,
        nloop=ALL_nloop,
    )


def short_name(path):
    return path.split("/")[-1].replace("new_results_", "").replace(".npy", "")


METRICS = [
    ("acc (FAQ)",      "acc",     "{:.4f}"),
    ("edges (FAQ)",    "qap",     "{:.1f}"),
    ("acc (LAP)",      "acc_p",   "{:.4f}"),
    ("edges (LAP)",    "qap_p",   "{:.1f}"),
    ("acc (argmax)",   "acc_max", "{:.4f}"),
    ("nloop",          "nloop",   "{:.1f}"),
]


def _header_sep(noises):
    noise_cols = " | ".join(f"  {n:.2f}  " for n in noises)
    sep_cols   = " | ".join("---------" for _ in noises)
    return f"| metric | {noise_cols} |", f"|:-------|{sep_cols}|"


def print_table(label, d):
    noises = d["noises"]
    header, sep = _header_sep(noises)
    print(f"\n**{label}**\n")
    print(header)
    print(sep)
    for name, key, fmt in METRICS:
        vals = " | ".join(fmt.format(d[key][i].mean()) for i in range(len(noises)))
        print(f"| {name} | {vals} |")


def print_comparison(datasets):
    noises = datasets[0][1]["noises"]
    header, sep = _header_sep(noises)
    for name, key, fmt in METRICS:
        print(f"\n**{name}**\n")
        print(header)
        print(sep)
        for path, d in datasets:
            vals = " | ".join(fmt.format(d[key][i].mean()) for i in range(len(noises)))
            print(f"| {short_name(path)} | {vals} |")


def make_table(paths):
    datasets = [(p, load_results(p)) for p in paths]
    if len(datasets) == 1:
        path, d = datasets[0]
        print_table(short_name(path), d)
    else:
        print_comparison(datasets)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    make_table(sys.argv[1:])
