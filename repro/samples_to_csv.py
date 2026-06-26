#!/usr/bin/env python3
"""Flatten a reproduction JSONL into a tidy long CSV — one row per
(cell, method, sample, acc, nce). Sample index `i` is the *same* graph pair
across every method, so this is set up for paired cross-method analysis, e.g.::

    import pandas as pd
    df = pd.read_csv("samples.csv")
    w = df.pivot_table(index=["cell", "sample"], columns="method", values="acc")
    # which pairs does ChFGNN solve that FAQ(D_cx) misses, and vice-versa?
    w[(w["chfgnn_faq"] > 0.9) & (w["faq_dcx"] < 0.1)]

Usage::

    python repro/samples_to_csv.py [results.jsonl] [-o samples.csv]   # default: stdout
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "results", "repro_seed0.jsonl")
FIELDS = ["table", "cell", "seed", "method", "sample", "acc", "nce"]


def rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            methods = r.get("methods")
            if not methods:  # old means-only schema carries no per-sample data
                continue
            cell = r.get("cell") or f"{r['family']}@{r['noise']}"
            for name, m in methods.items():
                accs, nces = m.get("acc"), m["nce"]
                for i in range(len(nces)):
                    yield {
                        "table": r["table"], "cell": cell, "seed": r.get("seed"),
                        "method": name, "sample": i,
                        "acc": "" if accs is None else accs[i],
                        "nce": nces[i],
                    }


def main():
    ap = argparse.ArgumentParser(description="Reproduction JSONL -> tidy per-sample CSV.")
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("-o", "--out", default="-", help="Output CSV (default: stdout).")
    args = ap.parse_args()
    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    try:
        w = csv.DictWriter(out, fieldnames=FIELDS)
        w.writeheader()
        n = 0
        for row in rows(args.path):
            w.writerow(row)
            n += 1
    finally:
        if out is not sys.stdout:
            out.close()
    if not n:
        sys.stderr.write("No per-sample data found (means-only run?). Re-run reproduce_results.py.\n")


if __name__ == "__main__":
    main()
